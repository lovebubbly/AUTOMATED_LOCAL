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
        
        # Initialize model
        if self.model_name == "realesr-animevideov3":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=None,  # Will auto-download
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
