"""
Video Upscaler - Real-ESRGAN based video upscaling module

720p → 1440p 비디오 업스케일을 위한 모듈입니다.
Real-ESRGAN을 사용하여 프레임별로 업스케일 후 재조합합니다.
"""

import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoUpscaler:
    """
    Real-ESRGAN 기반 비디오 업스케일러.
    
    720p (1280x720) → 1440p (2560x1440) 업스케일을 수행합니다.
    """
    
    def __init__(
        self,
        model_name: str = "realesr-animevideov3",
        scale: int = 2,
        gpu_id: int = 0
    ):
        """
        Args:
            model_name: Real-ESRGAN 모델명
                - "realesr-animevideov3" (애니메이션/AI 생성 영상에 최적)
                - "RealESRGAN_x4plus" (실사 영상)
                - "RealESRGAN_x4plus_anime_6B" (애니메이션)
            scale: 업스케일 배율 (2 = 720p→1440p, 4 = 720p→4K)
            gpu_id: 사용할 GPU ID
        """
        self.model_name = model_name
        self.scale = scale
        self.gpu_id = gpu_id
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """필수 의존성 확인."""
        # Check for realesrgan-ncnn-vulkan or Python realesrgan
        self.use_ncnn = shutil.which("realesrgan-ncnn-vulkan") is not None
        
        if not self.use_ncnn:
            try:
                from realesrgan import RealESRGANer
                self.use_python = True
            except ImportError:
                self.use_python = False
                logger.warning(
                    "Real-ESRGAN not found. Install via:\n"
                    "  pip install realesrgan basicsr\n"
                    "Or download realesrgan-ncnn-vulkan from:\n"
                    "  https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases"
                )
        else:
            self.use_python = False
    
    def upscale_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        비디오를 업스케일합니다.
        
        Args:
            input_path: 입력 비디오 경로
            output_path: 출력 비디오 경로 (None이면 자동 생성)
            progress_callback: 진행률 콜백 (current_frame, total_frames)
            
        Returns:
            출력 비디오 경로
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        if output_path is None:
            stem = input_path.stem
            suffix = input_path.suffix
            output_path = input_path.parent / f"{stem}_1440p{suffix}"
        else:
            output_path = Path(output_path)
        
        logger.info(f"🎬 Upscaling video: {input_path}")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Scale: {self.scale}x, Model: {self.model_name}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir = Path(temp_dir) / "frames"
            upscaled_dir = Path(temp_dir) / "upscaled"
            frames_dir.mkdir()
            upscaled_dir.mkdir()
            
            # Step 1: Extract frames
            logger.info("📤 Extracting frames...")
            fps = self._extract_frames(input_path, frames_dir)
            
            # Step 2: Upscale frames
            logger.info("🔄 Upscaling frames...")
            frame_count = len(list(frames_dir.glob("*.png")))
            self._upscale_frames(frames_dir, upscaled_dir, progress_callback, frame_count)
            
            # Step 3: Reassemble video
            logger.info("📥 Reassembling video...")
            self._reassemble_video(upscaled_dir, output_path, fps, input_path)
        
        logger.info(f"✅ Upscaling complete: {output_path}")
        return str(output_path)
    
    def _extract_frames(self, video_path: Path, output_dir: Path) -> float:
        """FFmpeg로 프레임 추출."""
        # Get video FPS
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            str(video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        fps_str = result.stdout.strip()
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            fps = num / den
        else:
            fps = float(fps_str) if fps_str else 30.0
        
        # Extract frames
        extract_cmd = [
            "ffmpeg", "-i", str(video_path),
            "-qscale:v", "2",
            str(output_dir / "frame_%06d.png")
        ]
        subprocess.run(extract_cmd, capture_output=True, check=True)
        
        return fps
    
    def _upscale_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable],
        total_frames: int
    ) -> None:
        """프레임 업스케일."""
        if self.use_ncnn:
            # Use realesrgan-ncnn-vulkan
            cmd = [
                "realesrgan-ncnn-vulkan",
                "-i", str(input_dir),
                "-o", str(output_dir),
                "-n", self.model_name,
                "-s", str(self.scale),
                "-g", str(self.gpu_id),
                "-f", "png"
            ]
            subprocess.run(cmd, check=True)
        elif self.use_python:
            self._upscale_frames_python(input_dir, output_dir, progress_callback, total_frames)
        else:
            raise RuntimeError("No Real-ESRGAN backend available")
    
    def _upscale_frames_python(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable],
        total_frames: int
    ) -> None:
        """Python Real-ESRGAN으로 프레임 업스케일."""
        import cv2
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        
        # Initialize model - architecture must match the pretrained weights
        if self.model_name == "realesr-animevideov3":
            # realesr-animevideov3 uses SRVGGNetCompact (VGG-style, NOT RRDBNet!)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        elif self.model_name == "realesrgan-x4plus-anime":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        else:
            # Default: RealESRGAN_x4plus (23 blocks)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        
        logger.info(f"   Loading model: {self.model_name} (native scale: {netscale}x)")
        
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_url,  # Auto-download from URL
            model=model,
            half=True,
            gpu_id=self.gpu_id
        )
        
        frames = sorted(input_dir.glob("*.png"))
        for i, frame_path in enumerate(frames):
            img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            output, _ = upsampler.enhance(img, outscale=self.scale)
            
            output_path = output_dir / frame_path.name
            cv2.imwrite(str(output_path), output)
            
            if progress_callback:
                progress_callback(i + 1, total_frames)
    
    def _reassemble_video(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: float,
        original_video: Path
    ) -> None:
        """FFmpeg로 비디오 재조합."""
        # Check if original has audio
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(original_video)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        has_audio = "audio" in result.stdout
        
        if has_audio:
            # With audio
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%06d.png"),
                "-i", str(original_video),
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(output_path)
            ]
        else:
            # No audio
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%06d.png"),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]
        
        subprocess.run(cmd, capture_output=True, check=True)


def upscale_video(
    input_path: str,
    output_path: Optional[str] = None,
    scale: int = 2,
    model: str = "realesr-animevideov3"
) -> str:
    """
    비디오 업스케일 간편 함수.
    
    Args:
        input_path: 입력 비디오 경로
        output_path: 출력 비디오 경로 (None이면 자동 생성)
        scale: 업스케일 배율 (2 = 720p→1440p)
        model: Real-ESRGAN 모델명
        
    Returns:
        출력 비디오 경로
    """
    upscaler = VideoUpscaler(model_name=model, scale=scale)
    return upscaler.upscale_video(input_path, output_path)


# CLI support
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python upscale.py <input_video> [output_video]")
        print("       Upscales 720p video to 1440p using Real-ESRGAN")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = upscale_video(input_video, output_video)
        print(f"✅ Output saved to: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


# ============================================================
# VIDEO CONCATENATION
# ============================================================

class VideoConcatenator:
    """
    여러 비디오 파일을 하나로 이어붙이는 클래스.
    
    FFmpeg의 concat demuxer를 사용하여 무손실 연결합니다.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Args:
            output_dir: 출력 디렉토리 (None이면 첫 번째 입력 파일의 디렉토리 사용)
        """
        self.output_dir = Path(output_dir) if output_dir else None
    
    def concatenate(
        self,
        video_paths: list,
        output_filename: str = "final_output.mp4",
        reencode: bool = False
    ) -> str:
        """
        여러 비디오를 하나로 연결합니다.
        
        Args:
            video_paths: 비디오 파일 경로 리스트 (순서대로 연결)
            output_filename: 출력 파일명
            reencode: True면 재인코딩 (다른 코덱/해상도 비디오 연결 시)
            
        Returns:
            출력 파일 경로
        """
        if len(video_paths) < 2:
            raise ValueError("At least 2 videos required for concatenation")
        
        # Validate all files exist
        video_paths = [Path(p) for p in video_paths]
        for vp in video_paths:
            if not vp.exists():
                raise FileNotFoundError(f"Video not found: {vp}")
        
        # Determine output directory
        if self.output_dir:
            output_dir = self.output_dir
        else:
            output_dir = video_paths[0].parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        
        logger.info(f"🎬 Concatenating {len(video_paths)} videos...")
        
        if reencode:
            self._concat_with_reencode(video_paths, output_path)
        else:
            self._concat_demuxer(video_paths, output_path)
        
        logger.info(f"✅ Concatenation complete: {output_path}")
        return str(output_path)
    
    def _concat_demuxer(self, video_paths: list, output_path: Path) -> None:
        """FFmpeg concat demuxer 사용 (무손실, 같은 코덱 필요)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for vp in video_paths:
                # Escape single quotes in path
                escaped_path = str(vp.absolute()).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
            concat_list = f.name
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                str(output_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        finally:
            os.unlink(concat_list)
    
    def _concat_with_reencode(self, video_paths: list, output_path: Path) -> None:
        """재인코딩하면서 연결 (다른 코덱/해상도 지원)."""
        # Check if videos have audio
        has_audio = self._check_video_has_audio(video_paths[0])
        
        # Build filter complex for concat
        filter_complex = ""
        for i in range(len(video_paths)):
            if has_audio:
                filter_complex += f"[{i}:v][{i}:a]"
            else:
                filter_complex += f"[{i}:v]"
        
        if has_audio:
            filter_complex += f"concat=n={len(video_paths)}:v=1:a=1[outv][outa]"
        else:
            filter_complex += f"concat=n={len(video_paths)}:v=1:a=0[outv]"
        
        cmd = [
            "ffmpeg", "-y",
        ]
        for vp in video_paths:
            cmd.extend(["-i", str(vp)])
        
        if has_audio:
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[outv]",
                "-map", "[outa]",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                str(output_path)
            ])
        else:
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[outv]",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                str(output_path)
            ])
        subprocess.run(cmd, capture_output=True, check=True)
    
    def _check_video_has_audio(self, video_path: Path) -> bool:
        """비디오에 오디오 스트림이 있는지 확인."""
        try:
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(video_path)
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            return "audio" in result.stdout
        except Exception:
            return False
    
    def concatenate_directory(
        self,
        directory: str,
        pattern: str = "*.mp4",
        output_filename: str = "final_output.mp4",
        sort_by: str = "name"
    ) -> str:
        """
        디렉토리 내 모든 비디오를 연결합니다.
        
        Args:
            directory: 비디오가 있는 디렉토리
            pattern: 파일 패턴 (예: "*.mp4", "block_*.mp4")
            output_filename: 출력 파일명
            sort_by: 정렬 기준 ("name" 또는 "time")
            
        Returns:
            출력 파일 경로
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        videos = list(dir_path.glob(pattern))
        if not videos:
            raise ValueError(f"No videos matching '{pattern}' in {directory}")
        
        # Sort videos
        if sort_by == "name":
            videos.sort(key=lambda x: x.name)
        elif sort_by == "time":
            videos.sort(key=lambda x: x.stat().st_mtime)
        else:
            raise ValueError(f"Invalid sort_by: {sort_by}")
        
        logger.info(f"📂 Found {len(videos)} videos in {directory}")
        for i, v in enumerate(videos):
            logger.info(f"   {i+1}. {v.name}")
        
        return self.concatenate(videos, output_filename)


def concat_videos(
    video_paths: list,
    output_path: Optional[str] = None,
    reencode: bool = False
) -> str:
    """
    비디오 연결 간편 함수.
    
    Args:
        video_paths: 비디오 파일 경로 리스트
        output_path: 출력 경로 (None이면 자동 생성)
        reencode: 재인코딩 여부
        
    Returns:
        출력 파일 경로
    """
    output_filename = Path(output_path).name if output_path else "final_output.mp4"
    output_dir = Path(output_path).parent if output_path else None
    
    concat = VideoConcatenator(output_dir=str(output_dir) if output_dir else None)
    return concat.concatenate(video_paths, output_filename, reencode)


def concat_directory(
    directory: str,
    pattern: str = "*.mp4",
    output_filename: str = "final_output.mp4"
) -> str:
    """
    디렉토리 내 비디오 연결 간편 함수.
    
    Args:
        directory: 비디오 디렉토리
        pattern: 파일 패턴
        output_filename: 출력 파일명
        
    Returns:
        출력 파일 경로
    """
    concat = VideoConcatenator()
    return concat.concatenate_directory(directory, pattern, output_filename)


# ============================================================
# SMART CONCATENATOR - Scene-aware transitions
# ============================================================

class SmartConcatenator:
    """
    스마트 비디오 연결기 - CSV 분석하여 새 씬에만 트랜지션 적용.
    
    Production Table CSV를 분석하여:
    - [Input: Last Frame]로 시작하는 블록: 트랜지션 없이 연결 (연속 씬)
    - 새 프롬프트로 시작하는 블록: 크로스페이드 트랜지션 적용 (새 씬)
    """
    
    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        transition_duration: float = 0.3,
        transition_type: str = "crossfade"
    ):
        """
        Args:
            csv_path: Production Table CSV 경로
            video_dir: 비디오 파일 디렉토리
            transition_duration: 트랜지션 길이 (초)
            transition_type: 트랜지션 타입 ("crossfade", "fade_black")
        """
        self.csv_path = Path(csv_path)
        self.video_dir = Path(video_dir)
        self.transition_duration = transition_duration
        self.transition_type = transition_type
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
    
    def analyze_transitions(self) -> list:
        """
        CSV를 분석하여 트랜지션이 필요한 블록을 찾습니다.
        
        Returns:
            list of tuples: [(block_id, needs_transition), ...]
        """
        import pandas as pd
        
        df = pd.read_csv(self.csv_path)
        transitions = []
        
        for index, row in df.iterrows():
            block_id = str(row['Block']).zfill(2)
            start_frame_prompt = str(row.get('Nano Banana (Start Frame)', ''))
            section = str(row.get('Section', ''))
            
            # 이전 블록의 섹션 (있다면)
            prev_section = df.iloc[index - 1]['Section'] if index > 0 else None
            
            # 트랜지션이 필요한 조건:
            # 1. [Input: Last Frame]로 시작하지 않음 (새 씬)
            # 2. 섹션이 변경됨 (Intro → Verse 등)
            # 3. 첫 블록이 아님
            is_continuation = "[Input: Last Frame" in start_frame_prompt or "[Loop Bank" in start_frame_prompt
            section_changed = prev_section is not None and section != prev_section
            
            needs_transition = False
            if index > 0:  # 첫 블록은 트랜지션 불필요
                if not is_continuation or section_changed:
                    needs_transition = True
            
            transitions.append({
                'block_id': block_id,
                'needs_transition': needs_transition,
                'section': section,
                'reason': 'new_scene' if not is_continuation else ('section_change' if section_changed else 'continuation')
            })
            
            logger.debug(f"Block {block_id}: transition={needs_transition}, reason={transitions[-1]['reason']}")
        
        return transitions
    
    def concatenate_smart(
        self,
        output_filename: str = "final_smart.mp4",
        video_pattern: str = "block_{block_id}_video.mp4"
    ) -> str:
        """
        스마트 트랜지션으로 비디오를 연결합니다.
        
        Args:
            output_filename: 출력 파일명
            video_pattern: 비디오 파일 패턴 ({block_id}는 자동 치환)
            
        Returns:
            출력 파일 경로
        """
        transitions = self.analyze_transitions()
        
        # 존재하는 비디오 파일 찾기
        video_segments = []
        for t in transitions:
            video_name = video_pattern.format(block_id=t['block_id'])
            video_path = self.video_dir / video_name
            if video_path.exists():
                video_segments.append({
                    'path': video_path,
                    'block_id': t['block_id'],
                    'needs_transition': t['needs_transition'],
                    'reason': t['reason']
                })
            else:
                logger.warning(f"Video not found: {video_path}")
        
        if not video_segments:
            raise ValueError("No video files found")
        
        logger.info(f"🎬 Smart concatenation: {len(video_segments)} videos")
        for seg in video_segments:
            trans_mark = "🔀" if seg['needs_transition'] else "➡️"
            logger.info(f"   {trans_mark} Block {seg['block_id']} ({seg['reason']})")
        
        # 트랜지션 필요한 부분 카운트
        trans_count = sum(1 for seg in video_segments if seg['needs_transition'])
        logger.info(f"📊 {trans_count} transitions will be applied")
        
        # FFmpeg로 스마트 연결
        output_path = self.video_dir / output_filename
        self._concatenate_with_transitions(video_segments, output_path)
        
        logger.info(f"✅ Smart output saved: {output_path}")
        return str(output_path)
    
    def _concatenate_with_transitions(self, segments: list, output_path: Path) -> None:
        """
        FFmpeg로 트랜지션 포함 연결 (2단계 방식).
        
        1단계: 연속 씬(needs_transition=False)끼리 그룹으로 concat
        2단계: 그룹 사이에 xfade 트랜지션 적용
        """
        if len(segments) == 1:
            shutil.copy(segments[0]['path'], output_path)
            return
        
        import tempfile
        
        # 그룹 분할: 트랜지션이 필요한 지점에서 분할
        groups = []
        current_group = [segments[0]]
        
        for i in range(1, len(segments)):
            if segments[i]['needs_transition']:
                # 트랜지션 필요 = 새 그룹 시작
                groups.append(current_group)
                current_group = [segments[i]]
            else:
                # 연속 씬 = 현재 그룹에 추가
                current_group.append(segments[i])
        
        groups.append(current_group)  # 마지막 그룹 추가
        
        logger.info(f"📦 Split into {len(groups)} groups for processing")
        
        # 1단계: 각 그룹을 하나의 비디오로 concat
        temp_dir = Path(tempfile.mkdtemp())
        group_videos = []
        
        try:
            for idx, group in enumerate(groups):
                if len(group) == 1:
                    # 단일 비디오면 그대로 사용
                    group_videos.append(group[0]['path'])
                    logger.debug(f"   Group {idx+1}: 1 video (pass-through)")
                else:
                    # 여러 비디오면 concat
                    group_output = temp_dir / f"group_{idx}.mp4"
                    self._simple_concat(group, group_output)
                    group_videos.append(group_output)
                    logger.debug(f"   Group {idx+1}: {len(group)} videos -> {group_output.name}")
            
            # 2단계: 그룹 간 트랜지션 적용
            if len(group_videos) == 1:
                shutil.copy(group_videos[0], output_path)
            elif self.transition_type == "fade_to_black":
                # Fade to Black: 길이가 줄지 않음
                self._fade_to_black_groups(group_videos, output_path)
            else:
                # xfade: 중첩되어 길이가 줄어듦
                self._xfade_groups(group_videos, output_path)
        
        finally:
            # 임시 파일 정리
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _simple_concat(self, segments: list, output_path: Path) -> None:
        """단순 concat (연속 씬용)."""
        # concat demuxer 방식 사용
        list_file = output_path.parent / f"{output_path.stem}_list.txt"
        
        try:
            with open(list_file, 'w', encoding='utf-8') as f:
                for seg in segments:
                    # 절대 경로로 변환하고 백슬래시를 슬래시로 변경
                    abs_path = Path(seg['path']).resolve()
                    path_str = str(abs_path).replace('\\', '/')
                    f.write(f"file '{path_str}'\n")
            
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                logger.error(f"Concat error: {result.stderr.decode()}")
                # 폴백: 재인코딩 방식
                self._concat_with_reencode(segments, output_path)
        finally:
            if list_file.exists():
                list_file.unlink()
    
    def _concat_with_reencode(self, segments: list, output_path: Path) -> None:
        """재인코딩 방식 concat (폴백용)."""
        inputs = []
        for seg in segments:
            inputs.extend(["-i", str(seg['path'])])
        
        filter_str = "".join([f"[{i}:v]" for i in range(len(segments))]) + f"concat=n={len(segments)}:v=1:a=0[outv]"
        
        cmd = ["ffmpeg", "-y"]
        cmd.extend(inputs)
        cmd.extend([
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            str(output_path)
        ])
        subprocess.run(cmd, capture_output=True, check=True)
    
    def _fade_to_black_groups(self, group_videos: list, output_path: Path) -> None:
        """
        Fade to Black 방식 트랜지션.
        
        각 그룹의 끝에 fadeout, 시작에 fadein 적용.
        중첩 없이 concat하므로 길이가 줄지 않음.
        """
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            processed_videos = []
            fade_duration = self.transition_duration
            
            for i, video in enumerate(group_videos):
                # 비디오 길이 측정
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    str(video)
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True)
                try:
                    duration = float(result.stdout.strip())
                except:
                    duration = 5.0
                
                output_temp = temp_dir / f"faded_{i}.mp4"
                
                # 첫 번째가 아니면 fadein, 마지막이 아니면 fadeout
                filters = []
                
                if i > 0:
                    # Fade in from black
                    filters.append(f"fade=t=in:st=0:d={fade_duration}")
                
                if i < len(group_videos) - 1:
                    # Fade out to black
                    fadeout_start = max(0, duration - fade_duration)
                    filters.append(f"fade=t=out:st={fadeout_start:.2f}:d={fade_duration}")
                
                if filters:
                    filter_str = ",".join(filters)
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(video),
                        "-vf", filter_str,
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "18",
                        str(output_temp)
                    ]
                    subprocess.run(cmd, capture_output=True, check=True)
                    processed_videos.append(output_temp)
                else:
                    # 필터 없으면 그대로 사용
                    processed_videos.append(video)
            
            # Concat demuxer로 연결
            list_file = temp_dir / "fade_list.txt"
            with open(list_file, 'w', encoding='utf-8') as f:
                for v in processed_videos:
                    path_str = str(Path(v).resolve()).replace('\\', '/')
                    f.write(f"file '{path_str}'\n")
            
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode != 0:
                logger.error(f"Fade concat error: {result.stderr.decode()}")
                # 폴백: 재인코딩 방식
                inputs = []
                for v in processed_videos:
                    inputs.extend(["-i", str(v)])
                
                filter_str = "".join([f"[{i}:v]" for i in range(len(processed_videos))]) + f"concat=n={len(processed_videos)}:v=1:a=0[outv]"
                
                cmd2 = ["ffmpeg", "-y"]
                cmd2.extend(inputs)
                cmd2.extend([
                    "-filter_complex", filter_str,
                    "-map", "[outv]",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    str(output_path)
                ])
                subprocess.run(cmd2, capture_output=True, check=True)
            
            logger.info(f"✅ Fade to Black transition applied ({len(group_videos)} groups)")
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _xfade_groups(self, group_videos: list, output_path: Path) -> None:
        """그룹 간 xfade 적용."""
        
        def get_video_duration(path):
            try:
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    str(path)
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True)
                return float(result.stdout.strip())
            except:
                return 5.0
        
        inputs = []
        filter_parts = []
        
        for i, video in enumerate(group_videos):
            inputs.extend(["-i", str(video)])
        
        # 모든 입력 정규화
        normalized = []
        for i in range(len(group_videos)):
            filter_parts.append(f"[{i}:v]fps=16,settb=AVTB[n{i}]")
            normalized.append(f"[n{i}]")
        
        # 순차적 xfade
        current_label = normalized[0]
        accumulated_duration = get_video_duration(group_videos[0])
        
        for i in range(1, len(group_videos)):
            next_input = normalized[i]
            out_label = f"[v{i}]"
            video_duration = get_video_duration(group_videos[i])
            
            offset = max(0, accumulated_duration - self.transition_duration)
            
            filter_parts.append(
                f"{current_label}{next_input}xfade=transition=fade:duration={self.transition_duration}:offset={offset:.2f}{out_label}"
            )
            
            accumulated_duration = offset + video_duration
            current_label = out_label
        
        filter_complex = ";".join(filter_parts)
        
        cmd = ["ffmpeg", "-y"]
        cmd.extend(inputs)
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", current_label,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            str(output_path)
        ])
        
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg xfade error: {result.stderr.decode()}")
            # 폴백: 단순 concat
            logger.info("Falling back to simple concatenation...")
            concat = VideoConcatenator(output_dir=str(output_path.parent))
            concat.concatenate([Path(v) for v in group_videos], output_path.name, reencode=True)


def smart_concat(
    csv_path: str,
    video_dir: str,
    output_filename: str = "final_smart.mp4",
    transition_duration: float = 0.3,
    transition_type: str = "crossfade"
) -> str:
    """
    스마트 트랜지션 연결 간편 함수.
    
    CSV를 분석하여 새 씬에만 트랜지션을 적용합니다.
    
    Args:
        csv_path: Production Table CSV 경로
        video_dir: 비디오 파일 디렉토리
        output_filename: 출력 파일명
        transition_duration: 트랜지션 길이 (초)
        transition_type: "crossfade" (중첩, 길이 줄어듦) 또는 "fade_to_black" (길이 유지)
        
    Returns:
        출력 파일 경로
        
    Example:
        >>> smart_concat(
        ...     csv_path="assets/production_table.csv",
        ...     video_dir="assets/images",
        ...     output_filename="final_music_video.mp4",
        ...     transition_type="fade_to_black"  # 길이 유지
        ... )
    """
    smart = SmartConcatenator(
        csv_path=csv_path,
        video_dir=video_dir,
        transition_duration=transition_duration,
        transition_type=transition_type
    )
    return smart.concatenate_smart(output_filename)

