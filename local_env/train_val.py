from ultralytics import YOLO

# Path to your config file
data_config = "../val/bdd_val.yaml"

# Load model (use yolov8n.pt for pretrained, or yolov8n.yaml for scratch)
model = YOLO("yolov8n.pt")

# Train for 2 epochs on val images
results = model.train(data=data_config, epochs=2)
