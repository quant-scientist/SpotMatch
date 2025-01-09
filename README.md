# YOLO-based Product Detection in Videos

This project uses YOLOv5 for robust product detection in video files. It identifies specific products or logos in video frames and outputs the timestamps and annotated frames where detections occur.

Features
	•	Detects specific objects/products/logos in videos using YOLOv5.
	•	Saves:
	•	Timestamps of detections in a text file.
	•	Annotated frames with bounding boxes and confidence scores.
	•	Easily configurable for different products or custom YOLO models.

Folder Structure

project/
├── detect_product.py         # Main detection script
├── utils.py                  # Helper functions
├── requirements.txt          # Required Python dependencies
├── input/                    # Input folder for videos and product images
│   ├── video.mp4             # Example video
│   ├── product.jpg           # (Optional) Example product image
├── output/                   # Output folder for results
│   ├── timestamps.txt        # Timestamps of detections
│   ├── detected_frames/      # Annotated frames
│   │   ├── frame_00001.jpg
│   │   ├── frame_00005.jpg

Installation

1. Clone the Repository

git clone https://github.com/your-repo/product-detection.git
cd product-detection

2. Install Dependencies

Make sure Python 3.8+ is installed, then install the required packages:

pip install -r requirements.txt

3. Install YOLOv5

Clone the YOLOv5 repository:

git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

Usage

1. Input Video and Model
	•	Place your input video (e.g., video.mp4) in the input/ folder.
	•	Use a pre-trained YOLOv5 model (e.g., yolov5s.pt) or train your custom YOLOv5 model for specific products/logos.

2. Run the Detection Script

Run the main script to process the video and detect products:

python detect_product.py

3. Output
	•	Timestamps: Saved in output/timestamps.txt.
	•	Annotated Frames: Saved in output/detected_frames/.

Customizing YOLOv5

Training a Custom Model

If detecting specific products or logos, train a custom YOLOv5 model:
	1.	Prepare Dataset:
	•	Collect images of the product/logo.
	•	Annotate the dataset using tools like LabelImg.
	2.	Train Model:

python train.py --img 640 --batch 16 --epochs 50 --data custom.yaml --weights yolov5s.pt


	3.	Use Custom Model:
Replace model_path in detect_product.py with the path to your trained weights (e.g., runs/train/exp/weights/best.pt).

Results
	1.	Timestamps:
	•	Saved in output/timestamps.txt:

5.30
12.75
20.10


	2.	Annotated Frames:
	•	Bounding boxes and labels are drawn on frames where products are detected:

output/detected_frames/frame_00005.jpg
output/detected_frames/frame_00010.jpg

Dependencies
	•	Python 3.8+
	•	Required Python libraries:
	•	PyTorch, Torchvision
	•	OpenCV
	•	Pandas, Matplotlib, PyYAML

Install dependencies using:

pip install -r requirements.txt

Future Improvements
	•	Add multi-object tracking for better temporal analysis.
	•	Implement support for real-time detection via webcam or live video streams.
	•	Enhance GUI for user-friendly interaction.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to contact Your Name for support or questions. Enjoy building with YOLO!

This README provides all necessary instructions for installation, usage, and customization while maintaining clarity. Let me know if you’d like additional enhancements!