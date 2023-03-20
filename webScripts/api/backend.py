from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CameraPosition(BaseModel):
    x: int
    y: int
    z: int
    snapid: int



@app.post("/get/splines")
async def root(camera_position: CameraPosition):
    
    return {"message": "Hello World"}




