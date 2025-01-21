import os
import cv2
import torch
import tempfile
from PIL import Image

def recognize_faces_in_video(video_stream, output_stream, embedding_data, mtcnn, resnet, threshold=0.7):
    """
    Recognize faces in a video and output a labeled video.
    This version writes the video to a temporary file before returning it as a byte stream.
    """
    # Create a temporary file for the video
    with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
        temp_video_file.write(video_stream.read())
        temp_video_file_path = temp_video_file.name

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(temp_video_file_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {temp_video_file_path}")
        os.remove(temp_video_file_path)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a temporary output file for the processed video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_video_file:
        temp_output_video_path = temp_output_video_file.name
        out = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break 

            frame_idx += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)

            # Detect faces and generate cropped tensors
            boxes, probs = mtcnn.detect(image)
            cropped_images = mtcnn(image)

            if boxes is not None:
                for box, prob, face in zip(boxes, probs, cropped_images):
                    if prob < 0.90:
                        continue

                    # Compare against known embeddings
                    emb = resnet(face.unsqueeze(0))
                    distances = {}
                    for known_emb, name in embedding_data:
                        dist = torch.dist(emb, known_emb).item()
                        distances[name] = dist

                    # Find the closest match
                    closest, min_dist = min(distances.items(), key=lambda x: x[1])

                    # Determine the label
                    name = closest if min_dist < threshold else "Unrecognized"
                    color = (0, 0, 255) if name == "Unrecognized" else (255, 0, 0)
                    label = f"{name} {min_dist:.2f}"

                    # Draw bounding box and label on the frame
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

            out.write(frame)

        cap.release()
        out.release()


    with open(temp_output_video_path, "rb") as f:
        video_bytes = f.read()

    
    os.remove(temp_video_file_path)  
    os.remove(temp_output_video_path) 

    # Write the video to the output stream
    output_stream.write(video_bytes)
    print(f"Labeled video saved to output stream")






