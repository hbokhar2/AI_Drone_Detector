from ultralytics import YOLO
import os

DATA_YAML_PATH = '../../drone_dataset/data.yaml'

MODEL_TYPE = 'yolov8n.pt'

NUM_EPOCHS = 25

PROJECT_DIR = 'TrainedAiFiles'
RUN_NAME = 'drone_run'

def train_model():
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: data.yaml not found at {DATA_YAML_PATH}")
        print("Please update DATA_YAML_PATH to the correct location.")
        return

    print(f"Loading base model: {MODEL_TYPE}")
    model = YOLO(MODEL_TYPE)

    print(f"Starting training for {NUM_EPOCHS} epochs...")

    results = model.train(
            data=DATA_YAML_PATH, 
            epochs=NUM_EPOCHS, 
            imgsz=416, 
            device=0,
            project=PROJECT_DIR,
            name=RUN_NAME
            )

    print("\n========================================================")
    print("Training finished!")
    print("Your trained model weights are saved in:")

    final_path = os.path.abspath(
            os.path.join(
                PROJECT_DIR, 
                RUN_NAME, 
                'weights', 
                'best.pt'
                )
            )

    print(final_path)
    print("\nIf you are using the default path, the file should be here:")
    print("/home/B0LD/Documents/Projects/Capstone/DroneDetection/AiTrainer/TrainedAiFiles/drone_run/weights/best.pt")
    print("========================================================\n")


if __name__ == "__main__":
    train_model()
