from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import threading
from automatic import Director

app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Director
director = Director()

# API Models
class StartRequest(BaseModel):
    mode: str = "full"  # "image", "video", "full"
    start_block: int = 1 # Resume from specific block

# Routes
@app.get("/api/status")
def get_status():
    return {
        "is_running": director.is_running,
        "status": director.status,
        "current_block": director.current_block,
        "logs": director.logs[-50:]  # Return last 50 logs
    }

@app.post("/api/start")
def start_automation(req: StartRequest):
    print(f"📥 API Request: Mode={req.mode}, StartBlock={req.start_block} (Type: {type(req.start_block)})")
    if director.is_running:
        raise HTTPException(status_code=400, detail="Director is already running")
    
    success = director.start(req.mode, req.start_block)
    if not success:
         raise HTTPException(status_code=500, detail="Failed to start")
    
    return {"message": f"Started in {req.mode} mode"}

@app.post("/api/stop")
def stop_automation():
    director.stop()
    return {"message": "Stop requested"}

@app.get("/api/assets")
def list_assets():
    assets_dir = os.path.join(os.getcwd(), "assets", "images")
    if not os.path.exists(assets_dir):
        return []
    
    files = sorted(
        [f for f in os.listdir(assets_dir) if f.endswith(('.png', '.mp4'))],
        key=lambda x: os.path.getmtime(os.path.join(assets_dir, x)),
        reverse=True
    )
    return files

# Serve Static Files
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
