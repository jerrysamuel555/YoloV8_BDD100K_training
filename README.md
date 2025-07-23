# YoloV8_BDD100K_training

This repository provides a pipeline for training a YOLOv8 object detection model on the BDD100K dataset, including label conversion, configuration, and training scripts.

## Project Structure

- `local_env/bdd_json_to_yolo.py`: Converts BDD100K JSON labels to YOLO format.
- `local_env/train_val.py`: Trains YOLOv8 on the prepared dataset using Ultralytics YOLO.
- `local_env/yolov8n_scratch.yaml`: YOLOv8n model config for training from scratch.
- `local_env/yolov8n.pt`: Pretrained YOLOv8n weights (optional).
- `val/`: Contains images, YOLO labels, and dataset config (`bdd_val.yaml`).
- `yolov8env/`: Python virtual environment (recommended for reproducibility).

## Setup Instructions

1. **Clone the repository and set up the environment:**
   - Create and activate a Python virtual environment (e.g., using `venv` or `conda`).
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Prepare the Data:**
   - Copy your BDD100K images into the `val/images/` folder.
   - Copy your BDD100K JSON label files into the `val/labels/` folder.
   - Update paths in `bdd_json_to_yolo.py` if your structure differs.

3. **Convert BDD100K Labels to YOLO Format:**
   - Run the conversion script:
     ```bash
     python local_env/bdd_json_to_yolo.py
     ```
   - This will create YOLO-format label files in `val/labels_yolo/`.

4. **Check Dataset Config:**
   - `val/bdd_val.yaml` should look like:
     ```yaml
     path: ../val
     train: images
     val: images
     labels: labels_yolo
     nc: 10
     names: ["bike", "bus", "car", "motor", "person", "rider", "traffic light", "traffic sign", "train", "truck"]
     ```

5. **Train YOLOv8 Model:**
   - Edit `local_env/train_val.py` to select pretrained (`yolov8n.pt`) or scratch (`yolov8n_scratch.yaml`) model.
   - Run training:
     ```bash
     python local_env/train_val.py
     ```
   - Training results will be saved in `local_env/runs/detect/`.

## Scripts Overview

### bdd_json_to_yolo.py
Converts BDD100K JSON labels to YOLO format. For each image, it:
- Reads the JSON label.
- Extracts bounding boxes and class names.
- Normalizes coordinates to YOLO format.
- Writes a `.txt` file per image in `val/labels_yolo/`.

### train_val.py
Trains YOLOv8 using Ultralytics. Key steps:
- Loads model (pretrained or from scratch).
- Uses `bdd_val.yaml` for data config.
- Trains for specified epochs (default: 2).

## Customization

- To train from scratch, use `yolov8n_scratch.yaml` as the model config.
- To use pretrained weights, use `yolov8n.pt`.
- Adjust number of epochs and other parameters in `train_val.py` as needed.

## Notes on YAML Configuration

- The `yolov8n_scratch.yaml` file has been modified for experimentation. You may see changes in the model architecture, class definitions, or other parameters. Feel free to further adjust this file to test different model settings or training strategies.

## Example: Training Process Screenshot

Below is a screenshot showing the start of the YOLOv8 training process:

![Training Started](Screenshot%202025-07-23%20161943.png)

## References
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)