from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN, InceptionResnetV1
import io
import json
import numpy as np

from face_recognition import recognize_faces_in_video

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins= origins
)

# Initialize models
device = "cpu"
mtcnn = MTCNN(device=device, keep_all=True, min_face_size=60, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to("cpu")

# Load the embeddings data
with open("embeddings.json", "r") as f:
    embeddings_loaded = json.load(f)

# Convert back to NumPy arrays if needed
embedding_data = [(np.array(item["embedding"]), item["name"]) for item in embeddings_loaded]

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    html_file_path = Path(__file__).parent / "index.html" 
    return html_file_path.read_text()

@app.post("/process")
async def process_video(video: UploadFile = File(...)):
    try:
        # Read the uploaded video file
        video_data = await video.read()

        # Use io.BytesIO to simulate a file-like object in memory
        video_stream = io.BytesIO(video_data)

        # Set output in memory (not saving to disk)
        output_video_stream = io.BytesIO()

        # Call the face recognition function, modifying it to write to memory instead of a file
        recognize_faces_in_video(video_stream, output_video_stream, embedding_data, mtcnn, resnet)

        # Set the output video stream position to the start before sending
        output_video_stream.seek(0)

        # Return the processed video as a StreamingResponse
        return StreamingResponse(output_video_stream, media_type="video/mp4")
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

app = app



