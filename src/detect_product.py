import torch
import cv2
import os
from utils import create_directory, save_frame_with_detection, save_timestamps, format_timestamp

def detect_product_with_yolo(video_path, model_path, output_dir, confidence_threshold=0.5):
    """
    Detects product appearances in a video using YOLOv5.

    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the YOLOv5 model weights.
        output_dir (str): Directory to save results.
        confidence_threshold (float): Confidence threshold for detections.

    Returns:
        None: Saves detected timestamps and annotated frames in the output directory.
    """
    # Create output directories
    create_directory(output_dir)
    frames_dir = os.path.join(output_dir, "detected_frames")
    create_directory(frames_dir)

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

    # Open the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    timestamps = []

    print("Processing video with YOLO...")

    while True:
        ret, frame = video.read()
        if not ret:
            break  # End of video

        # Run YOLO detection
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # Extract detection results

        # Filter detections by confidence threshold
        for x1, y1, x2, y2, conf, cls in detections:
            if conf >= confidence_threshold:
                # Save timestamp for the first detection in the frame
                timestamp = format_timestamp(frame_number, fps)
                timestamps.append(timestamp)

                # Annotate and save the frame
                label = f"{results.names[int(cls)]} {conf:.2f}"
                save_frame_with_detection(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    label,
                    frame_number,
                    frames_dir
                )

                # Only record one timestamp per frame
                break

        frame_number += 1

    video.release()

    # Save timestamps to a file
    timestamps_file = os.path.join(output_dir, "timestamps.txt")
    save_timestamps(timestamps, timestamps_file)

    print(f"Processing complete. Results saved in {output_dir}")
    if timestamps:
        print(f"Detected timestamps (seconds): {timestamps}")
    else:
        print("No matches detected.")


if __name__ == "__main__":
    # Define input and output paths
    video_path = "input/video.mp4"  # Path to the input video
    model_path = "yolov5s.pt"  # Path to YOLOv5 model weights
    output_dir = "output"  # Directory to save results

    # Run YOLO detection
    detect_product_with_yolo(video_path, model_path, output_dir, confidence_threshold=0.5)