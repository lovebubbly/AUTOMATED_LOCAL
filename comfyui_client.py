"""
ComfyUI API Client Module
Handles HTTP API and WebSocket communication with ComfyUI server.
"""

import json
import uuid
import time
import requests
import websocket
from pathlib import Path
from typing import Optional, Callable


class ComfyUIClient:
    """
    ComfyUI 서버와의 통신을 담당하는 클라이언트
    
    Features:
    - Image upload to ComfyUI input folder
    - Workflow prompt queuing
    - WebSocket-based progress monitoring
    - Result image downloading
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        """
        Initialize ComfyUI client.
        
        Args:
            host: ComfyUI server host (default: 127.0.0.1)
            port: ComfyUI server port (default: 8188)
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client_id = str(uuid.uuid4())
        
    def test_connection(self) -> bool:
        """
        Test if ComfyUI server is reachable.
        
        Returns:
            True if server responds, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/system_stats", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def upload_image(self, file_path: str, subfolder: str = "", image_type: str = "input") -> str:
        """
        Upload a local image to ComfyUI server's input folder.
        
        Args:
            file_path: Path to the local image file
            subfolder: Optional subfolder within ComfyUI's input directory
            image_type: Type of upload (input, temp, output)
            
        Returns:
            Uploaded filename (as stored in ComfyUI)
            
        Raises:
            FileNotFoundError: If file_path doesn't exist
            RuntimeError: If upload fails
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")
        
        # Prepare multipart form data
        with open(file_path, 'rb') as f:
            files = {
                'image': (path.name, f, 'image/png')
            }
            data = {
                'type': image_type,
                'overwrite': 'true'
            }
            if subfolder:
                data['subfolder'] = subfolder
            
            response = requests.post(
                f"{self.base_url}/upload/image",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code != 200:
            raise RuntimeError(f"Upload failed: {response.status_code} - {response.text}")
        
        result = response.json()
        return result.get('name', path.name)
    
    def queue_prompt(self, workflow: dict) -> str:
        """
        Queue a workflow for execution.
        
        Args:
            workflow: Workflow dictionary (ComfyUI API format)
            
        Returns:
            prompt_id for tracking execution
            
        Raises:
            RuntimeError: If queuing fails
        """
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        response = requests.post(
            f"{self.base_url}/prompt",
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Queue prompt failed: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Check for validation errors
        if 'error' in result:
            error_info = result.get('node_errors', result.get('error', 'Unknown error'))
            raise RuntimeError(f"Workflow validation error: {error_info}")
        
        return result.get('prompt_id')
    
    def wait_for_completion(
        self, 
        prompt_id: str, 
        timeout: int = 600,
        log_callback: Optional[Callable[[str], None]] = None,
        reconnect_attempts: int = 3,
        reconnect_delay: int = 5
    ) -> Optional[dict]:
        """
        Wait for workflow execution to complete using WebSocket.
        
        Args:
            prompt_id: The prompt ID to monitor
            timeout: Maximum wait time in seconds
            log_callback: Optional callback for progress logging
            reconnect_attempts: Number of reconnection attempts on disconnect
            reconnect_delay: Delay between reconnection attempts
            
        Returns:
            Dictionary with output info: {"node_id": str, "filename": str, "subfolder": str}
            Returns None if execution fails or times out
        """
        ws = None
        attempts = 0
        start_time = time.time()
        output_images = None
        
        while attempts <= reconnect_attempts:
            try:
                ws = websocket.create_connection(
                    f"{self.ws_url}?clientId={self.client_id}",
                    timeout=10
                )
                
                if log_callback and attempts > 0:
                    log_callback(f"   🔄 WebSocket reconnected (attempt {attempts})")
                
                while True:
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        if log_callback:
                            log_callback(f"   ⚠️ Generation timed out after {timeout}s")
                        return None
                    
                    # Receive message with timeout
                    ws.settimeout(5.0)
                    try:
                        message = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    
                    if not message:
                        continue
                    
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    # Progress update
                    if msg_type == 'progress':
                        progress_data = data.get('data', {})
                        value = progress_data.get('value', 0)
                        max_val = progress_data.get('max', 1)
                        percent = int((value / max_val) * 100) if max_val > 0 else 0
                        if log_callback and percent % 10 == 0:  # Log every 10%
                            log_callback(f"   ⏳ Progress: {percent}%")
                    
                    # Execution started
                    elif msg_type == 'execution_start':
                        exec_data = data.get('data', {})
                        if exec_data.get('prompt_id') == prompt_id:
                            if log_callback:
                                log_callback(f"   ⚡ Execution started...")
                    
                    # Node executed - capture output
                    elif msg_type == 'executed':
                        exec_data = data.get('data', {})
                        if exec_data.get('prompt_id') == prompt_id:
                            output = exec_data.get('output') or {}
                            # Check for images (SaveImage) or gifs/videos (VHS_VideoCombine)
                            media_list = output.get('images') or output.get('gifs') or output.get('videos') or []
                            if media_list:
                                media_info = media_list[0]  # Take first item
                                output_images = {
                                    'node_id': exec_data.get('node'),
                                    'filename': media_info.get('filename'),
                                    'subfolder': media_info.get('subfolder', ''),
                                    'type': media_info.get('type', 'output')
                                }
                    
                    # Execution complete
                    elif msg_type == 'execution_success' or msg_type == 'executing':
                        exec_data = data.get('data', {})
                        # 'executing' with null node means execution finished
                        if msg_type == 'executing' and exec_data.get('node') is None:
                            if exec_data.get('prompt_id') == prompt_id:
                                if output_images:
                                    return output_images
                                # If no images captured yet, try to get history
                                return self._get_history_images(prompt_id)
                    
                    # Execution error
                    elif msg_type == 'execution_error':
                        exec_data = data.get('data', {})
                        if exec_data.get('prompt_id') == prompt_id:
                            error_msg = exec_data.get('exception_message', 'Unknown error')
                            if log_callback:
                                log_callback(f"   ❌ Execution error: {error_msg}")
                            return None
                            
            except websocket.WebSocketException as e:
                attempts += 1
                if log_callback:
                    log_callback(f"   ⚠️ WebSocket error: {e}")
                if attempts <= reconnect_attempts:
                    if log_callback:
                        log_callback(f"   🔄 Reconnecting in {reconnect_delay}s...")
                    time.sleep(reconnect_delay)
                else:
                    if log_callback:
                        log_callback(f"   ❌ WebSocket connection failed after {reconnect_attempts} attempts")
                    return None
            finally:
                if ws:
                    try:
                        ws.close()
                    except:
                        pass
        
        return None
    
    def _get_history_images(self, prompt_id: str) -> Optional[dict]:
        """
        Get output images from execution history.
        
        Args:
            prompt_id: The prompt ID to query
            
        Returns:
            Dictionary with image info or None
        """
        try:
            response = requests.get(
                f"{self.base_url}/history/{prompt_id}",
                timeout=10
            )
            
            if response.status_code != 200:
                return None
            
            history = response.json()
            prompt_history = history.get(prompt_id, {})
            outputs = prompt_history.get('outputs', {})
            
            # Find first node with images or gifs/videos
            for node_id, node_output in outputs.items():
                if not node_output:
                    continue
                # Check for images (SaveImage) or gifs/videos (VHS_VideoCombine)
                media_list = node_output.get('images') or node_output.get('gifs') or node_output.get('videos') or []
                if media_list:
                    media_info = media_list[0]
                    return {
                        'node_id': node_id,
                        'filename': media_info.get('filename'),
                        'subfolder': media_info.get('subfolder', ''),
                        'type': media_info.get('type', 'output')
                    }
            
            return None
            
        except requests.exceptions.RequestException:
            return None
    
    def download_image(
        self, 
        filename: str, 
        output_path: str,
        subfolder: str = "",
        image_type: str = "output"
    ) -> str:
        """
        Download a generated image from ComfyUI.
        
        Args:
            filename: Filename in ComfyUI's output
            output_path: Local path to save the image
            subfolder: Subfolder within ComfyUI's output directory
            image_type: Type of image (output, temp, input)
            
        Returns:
            Path to saved file
            
        Raises:
            RuntimeError: If download fails
        """
        params = {
            'filename': filename,
            'type': image_type
        }
        if subfolder:
            params['subfolder'] = subfolder
        
        response = requests.get(
            f"{self.base_url}/view",
            params=params,
            timeout=30
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Download failed: {response.status_code}")
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return output_path
    
    def get_queue_status(self) -> dict:
        """
        Get current queue status.
        
        Returns:
            Dictionary with queue_running and queue_pending counts
        """
        try:
            response = requests.get(f"{self.base_url}/queue", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'running': len(data.get('queue_running', [])),
                    'pending': len(data.get('queue_pending', []))
                }
        except requests.exceptions.RequestException:
            pass
        return {'running': 0, 'pending': 0}
    
    def interrupt_current(self) -> bool:
        """
        Interrupt currently running generation.
        
        Returns:
            True if interrupt was sent successfully
        """
        try:
            response = requests.post(f"{self.base_url}/interrupt", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


# Simple test when run directly
if __name__ == "__main__":
    print("Testing ComfyUI Client...")
    client = ComfyUIClient()
    
    if client.test_connection():
        print("✅ Connected to ComfyUI server")
        status = client.get_queue_status()
        print(f"   Queue: {status['running']} running, {status['pending']} pending")
    else:
        print("❌ Cannot connect to ComfyUI server at localhost:8188")
        print("   Make sure ComfyUI is running!")
