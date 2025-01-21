from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi import HTTPException
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1
import io
import torch
from fastapi.staticfiles import StaticFiles
from face_recognition import recognize_faces_in_video

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize models
device = "cpu"
mtcnn = MTCNN(device=device, keep_all=True, min_face_size=60, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to("cpu")

embedding_data  = torch.load("embeddings.pt")

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    html_file_path = Path(__file__).parent / "index.html" 
    return html_file_path.read_text()

from fastapi.responses import StreamingResponse

@app.post("/process")
async def process_video(video: UploadFile = File(...)):
    try:
        # Read the uploaded video file into bytes
        video_data = await video.read()

        # Use io.BytesIO to simulate a file-like object in memory
        video_stream = io.BytesIO(video_data)

        # Set output in memory (not saving to disk)
        output_video_stream = io.BytesIO()

        # Call the face recognition function, modifying it to accept the video as a byte stream
        recognize_faces_in_video(video_stream, output_video_stream, embedding_data, mtcnn, resnet)

        # Set the output video stream position to the start before sending
        output_video_stream.seek(0)

        # Return the processed video as a StreamingResponse with the correct MIME type
        return StreamingResponse(output_video_stream, media_type="video/mp4")
    
    except Exception as e:
        # Log the detailed error message
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
