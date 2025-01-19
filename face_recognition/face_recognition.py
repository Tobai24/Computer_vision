import io
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize MTCNN and InceptionResnetV1 models
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

embedding_data = torch.load("embeddings.pt")

mtcnn = MTCNN(device=device, keep_all=True, min_face_size=60, post_process=False)
resnet = InceptionResnetV1(pretrained="vggface2").eval()

def recognize_faces_in_video(video_data, output_stream, embedding_data, mtcnn, resnet, threshold=0.7):
    """
    Recognize faces in a video (from byte data) and output a labeled video to a byte stream.
    
    Args:
        video_data (bytes): Byte data of the input video file.
        output_stream (io.BytesIO): The byte stream where the output labeled video will be saved.
        embedding_data (list): List of tuples (embedding, name) for known faces.
        mtcnn: MTCNN face detector.
        resnet: Face recognition model.
        threshold (float): Distance threshold for recognition.
    """
    # Convert video data bytes to numpy array
    video_np = np.frombuffer(video_data, dtype=np.uint8)

    # Decode the video into individual frames using OpenCV
    cap = cv2.VideoCapture(cv2.imdecode(video_np, cv2.IMREAD_COLOR))
    if not cap.isOpened():
        print(f"Error: Cannot decode video")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video

    # Create a VideoWriter object to write the processed frames to the output byte stream
    out = cv2.VideoWriter(output_stream, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_idx += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
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

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Labeled video saved to output stream")
