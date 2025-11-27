from ultralytics import YOLO
import os

TRAINED_WEIGHTS_PATH = '../TrainedAiFiles/drone_run/weights/best.pt' 

IMG_SIZE = 416 

def export_model():
    print("Loading trained model from:", TRAINED_WEIGHTS_PATH)
    try:
        model = YOLO(TRAINED_WEIGHTS_PATH)

        print(f"Exporting model to ONNX format with image size {IMG_SIZE}x{IMG_SIZE}...")

        model.export(
                format='onnx', 
                imgsz=IMG_SIZE, 
                opset=17, 
                simplify=True
                )

        print("\n========================================================")
        print("ONNX Export Successful!")
        print("Model saved as: best.onnx (in drone_run)")
        print("========================================================\n")

    except FileNotFoundError:
        print(f"ERROR: Trained weights not found at {TRAINED_WEIGHTS_PATH}")
        print("Ensure the path is correct and the training run completed.")

if __name__ == "__main__":
    export_model()
