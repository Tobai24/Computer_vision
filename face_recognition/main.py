from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pathlib import Path
import torch
import shutil
import io

from face_recognition import recognize_faces_in_video, MTCNN, InceptionResnetV1

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    html_file_path = Path(__file__).parent / "index.html"  # Update the path to your HTML file
    return html_file_path.read_text()

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(device=device, keep_all=True, min_face_size=60, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the embeddings data
embedding_data = torch.load("embeddings.pt")

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



