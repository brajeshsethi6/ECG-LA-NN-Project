
import sys
import os
import torch
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.models.la_nn import LANN
from src.api.stream_engine import ECGStreamEngine
from contextlib import asynccontextmanager

# Global variables to store model
model_container = {"model": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    Config.ensure_dirs()
    model = LANN(Config)
    model_path = os.path.join(Config.MODELS_DIR, 'la_nn_best.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        print(f"✓ Model loaded from {model_path}")
    else:
        print("⚠ Model weights not found!")
        
    model = model.to(Config.DEVICE)
    model.eval()
    model_container["model"] = model
    
    yield
    # Shutdown logic
    model_container["model"] = None

app = FastAPI(title="LA-NN Real-time ECG Dashboard", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(frontend_dir, 'index.html'))

@app.get("/status")
def get_status():
    return {
        "status": "ready",
        "device": Config.DEVICE,
        "model": "LA-NN (Liquid Neural Network)"
    }

@app.websocket("/ws/ecg/{record_name}")
async def ecg_streaming(websocket: WebSocket, record_name: str):
    await websocket.accept()
    
    try:
        # Get model from container
        model = model_container["model"]
        if not model:
            await websocket.close(code=1001)
            return

        # Initialize engine
        stream_engine = ECGStreamEngine(model, record_name=record_name)
        
        # New optimized stream engine handles bundling and timing internally
        async for samples, prediction in stream_engine.stream_samples(bundle_size=15):
            data_packet = {
                "samples": samples,
                "prediction": prediction,
                "fs": 360
            }
            await websocket.send_text(json.dumps(data_packet))

    except WebSocketDisconnect:
        print(f"Client disconnected from {record_name}")
    except Exception as e:
        print(f"Error in streaming: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
