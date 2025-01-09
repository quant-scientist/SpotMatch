import os
import cv2


def create_directory(directory_path):
    """
    Creates a directory if it does not already exist.

    Args:
        directory_path (str): Path to the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def save_frame_with_detection(frame, match_top_left, match_bottom_right, label, frame_number, output_dir):
    """
    Saves a video frame with a bounding box and label indicating the detected product/logo.

    Args:
        frame (numpy.ndarray): The video frame to save.
        match_top_left (tuple): Top-left corner of the detected product/logo.
        match_bottom_right (tuple): Bottom-right corner of the detected product/logo.
        label (str): Detection label to annotate the frame.
        frame_number (int): Frame number being saved.
        output_dir (str): Directory to save the detected frame.
    """
    # Draw bounding box on the frame
    cv2.rectangle(frame, match_top_left, match_bottom_right, (0, 255, 0), 2)

    # Add label text
    cv2.putText(frame, label, (match_top_left[0], match_top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the frame
    frame_filename = os.path.join(output_dir, f"frame_{frame_number:05d}.jpg")
    cv2.imwrite(frame_filename, frame)


def save_timestamps(timestamps, output_file):
    """
    Saves a list of timestamps to a text file.

    Args:
        timestamps (list of float): List of timestamps to save.
        output_file (str): Path to the output file.
    """
    with open(output_file, "w") as f:
        for timestamp in timestamps:
            f.write(f"{timestamp:.2f}\n")


def format_timestamp(frame_number, fps):
    """
    Converts a frame number into a human-readable timestamp.

    Args:
        frame_number (int): Frame number in the video.
        fps (float): Frames per second of the video.

    Returns:
        float: Timestamp in seconds.
    """
    return frame_number / fps