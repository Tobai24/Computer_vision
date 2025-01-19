from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import torch
import shutil
import os

# Assuming your face recognition code is in the same file or imported
from face_recognition import recognize_faces_in_video, MTCNN, InceptionResnetV1

app = FastAPI()

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(device=device, keep_all=True, min_face_size=60, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the embeddings data
embedding_data = torch.load("embeddings.pt")

# Create an uploads directory if it doesn't exist
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/process")
async def process_video(video: UploadFile = File(...)):
    try:
        # Save the uploaded video
        video_path = UPLOAD_DIR / video.filename
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # Set output path for the processed video
        output_path = UPLOAD_DIR / f"processed_{video.filename}"

        # Call the face recognition function
        recognize_faces_in_video(str(video_path), str(output_path), embedding_data, mtcnn, resnet)

        return JSONResponse(content={"message": "Video processed successfully!", "output_video": str(output_path)}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

