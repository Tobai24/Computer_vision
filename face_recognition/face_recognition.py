import cv2
import numpy as np
from PIL import Image

def recognize_faces_in_video(video_data, output_stream, embedding_data, mtcnn, resnet, threshold=0.7):
    import io

    # Decode video from byte stream
    video_stream = io.BytesIO(video_data)
    cap = cv2.VideoCapture(video_stream)

    if not cap.isOpened():
        print("Error: Cannot decode video")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_stream, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        # Detect faces and probabilities
        boxes, probs = mtcnn.detect(image)
        cropped_images = mtcnn(image)

        if boxes is not None and cropped_images is not None:
            for box, prob, face in zip(boxes, probs, cropped_images):
                if face is None or prob < 0.90:
                    continue

                # Compare embeddings
                emb = resnet(face.unsqueeze(0))
                distances = {}
                for known_emb, name in embedding_data:
                    dist = np.linalg.norm(emb - known_emb)
                    distances[name] = dist


                closest, min_dist = min(distances.items(), key=lambda x: x[1])
                name = closest if min_dist < threshold else "Unrecognized"
                color = (0, 0, 255) if name == "Unrecognized" else (255, 0, 0)
                label = f"{name} {min_dist:.2f}"

                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write frame
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print("Labeled video saved to output stream")


