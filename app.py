from flask import Flask, jsonify, request
from ultralytics import YOLO
import threading
from pymongo import MongoClient
import shutil
import os
from bson.objectid import ObjectId
from dotenv import load_dotenv
import cv2

load_dotenv()

app = Flask(__name__)

# Global variable to track training status, progress of epochs and batches
training_status = {
    "status": "idle",
    "overall_progress": 0,  # Total training progress percentage
    "epoch_progress": 0,     # Current epoch progress percentage
    "current_epoch": 0,      # Current epoch number
    "total_epochs": 0        # Total number of epochs
}
stop_training_event = threading.Event()

mongo_url = os.getenv('MONGO_URL')
if(mongo_url is None):
    exit()
client = MongoClient(mongo_url)
db = client.deepsea

def progress_callback(trainer):
    """Callback to update training status and check stop signal."""
    global training_status, stop_training_event

    # Check if stop signal is set
    if stop_training_event.is_set():
        print("Training stop signal detected. Terminating training.")
        raise InterruptedError("Training stopped by user.")

    # Update progress details
    current_epoch = trainer.epoch if hasattr(trainer, 'epoch') else 0
    total_epochs = trainer.args.epochs if hasattr(trainer.args, 'epochs') else 1
    current_batch = trainer.batch if hasattr(trainer, 'batch') else 0
    total_batches = trainer.num_batches if hasattr(trainer, 'num_batches') else 1

    training_status["current_epoch"] = current_epoch + 1
    training_status["total_epochs"] = total_epochs
    training_status["overall_progress"] = int(((current_epoch + (current_batch / total_batches)) / total_epochs) * 100)
    training_status["epoch_progress"] = int((current_batch / total_batches) * 100)


def train_model(name, organizationId, datasetId, datasetName, modelId, modelType, modelPath):
    global training_status, stop_training_event

    stop_training_event.clear()

    training_status["status"] = "running"
    training_status["overall_progress"] = 0
    training_status["epoch_progress"] = 0

    try:
        new_model = db.models.insert_one({
            "name": name,
            "status": "training",
            "isPreTrained": False,
            "dataset": ObjectId(datasetId),
            "baseModel": ObjectId(modelId),
            "organization": ObjectId(organizationId),
            "type": modelType,
            "isActive": True
        })

        model = YOLO(modelPath)
        model.add_callback("on_train_batch_end", progress_callback)

        model.train(
            data="./data.yaml" if modelType != "classification" else "dataset",
            epochs=10,
            lr0=1e-4,
            imgsz=640,
            pretrained=True,
            project=f"result/{organizationId}",
            name=f"{datasetName}_{modelType}_{modelId}",
            exist_ok=True
        )

        new_model_path = f"./result/{organizationId}/{datasetName}_{modelType}_{modelId}/weights/best.pt"
        destination_path = f"{'../backend/' if os.getenv('STORAGE_PATH') == '' else os.getenv('STORAGE_PATH')}public/models/{organizationId}/{datasetName}_{modelType}_{modelId}.pt"
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy2(new_model_path, destination_path)

        db.models.update_one(
            {"_id": new_model.inserted_id},
            {"$set": {"status": "completed", "path": f"public/models/{organizationId}/{datasetName}_{modelType}_{modelId}.pt"}}
        )
    except InterruptedError:
        db.models.update_one(
            {"_id": new_model.inserted_id},
            {"$set": {"status": "stopped"}}
        )
        training_status["status"] = "stopped"
    except Exception as e:
        print(f"Error during training: {e}")
        db.models.update_one(
            {"_id": new_model.inserted_id},
            {"$set": {"status": "error"}}
        )
        training_status["status"] = "error"
    finally:
        shutil.rmtree("./dataset", ignore_errors=True)
        training_status["status"] = "idle"
        training_status["overall_progress"] = 0
        training_status["epoch_progress"] = 0
        training_status["total_epochs"] = 0
        training_status["current_epoch"] = 0

@app.route('/cancel', methods=['GET'])
def stop_training():
    global stop_training_event, training_status

    if training_status["status"] != "running":
        return jsonify({"msg": "No training process is running"}), 400

    stop_training_event.set()
    training_status["status"] = "stopping"

    return jsonify({"msg": "Training stop signal sent"}), 200

@app.route('/train', methods=['POST'])
def start_training():
    global training_status
    req_data = request.get_json()
    if training_status["status"] == "running":
        return jsonify({"msg": "Training already in progress"}), 400
    
    # Create a thread for training
    train_thread = threading.Thread(target=train_model, args=(req_data['name'],req_data['organizationId'], req_data['datasetId'], req_data['datasetName'], req_data['modelId'], req_data['modelType'], req_data['modelPath']))
    train_thread.daemon = True
    train_thread.start()
    
    return jsonify({"msg": "Training started"}), 200

@app.route('/status', methods=['GET'])
def check_training_status():
    return jsonify(training_status), 200

@app.route('/inference', methods=['POST'])
def inference_image():
    json_data = request.get_json()
    file = json_data['file']
    modelId = json_data['modelId']

    model = db.models.find_one({"_id":ObjectId(modelId)})
    if model is not None and model['status'] == "completed" and model['isActive'] == True and file is not None:
        print(model)
        print(file)
        modelPath = model['path']
        if model['type'] == "segmentation":
            yoloModel = YOLO(f"{os.getenv("STORAGE_PATH")}{modelPath}")
            results = yoloModel(file['path'])
            annotated_img = results[0].plot()
            cv2.imwrite(file['path'], cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
            print( "results",results)
            return jsonify({'file' : file}), 200
        
        if model['type'] == "objectdetection":
            print(f"{os.getenv("STORAGE_PATH")}{modelPath}")
            yoloModel = YOLO(f"{os.getenv("STORAGE_PATH")}{modelPath}")
            print(yoloModel)
            # yoloModel = YOLO(f"yolov8n.pt")
            results = yoloModel(file['path'])
            annotated_img = results[0].plot()
            cv2.imwrite(file['path'], cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
            print( "results",results)
            return jsonify({'file' : file}), 200
        
        if model['type'] == "classification":
            yoloModel = YOLO(f"{os.getenv("STORAGE_PATH")}{modelPath}")
            # model = YOLO("yolov8n-cls.pt")
            results = yoloModel(file['path'])
            print("results",results[0])
            probs = results[0].probs  # get classification probabilities
            top1_confidence = probs.top1
            classifications = results[0].names
            first_key = list(classifications.keys())[top1_confidence]
            first_value = classifications[first_key]
            return jsonify({'classname' : first_value}), 200

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
