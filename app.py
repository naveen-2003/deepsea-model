from flask import Flask, jsonify, request
from ultralytics import YOLO
import threading
from pymongo import MongoClient
import shutil
import os
from bson.objectid import ObjectId
from dotenv import load_dotenv

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

mongo_url = os.getenv('MONGO_URL')
if(mongo_url is None):
    exit()
client = MongoClient(mongo_url)
db = client.deepsea



def progress_callback(trainer):
    """Callback to update training status after each batch."""
    global training_status

    # Get the current state of training
    current_epoch = trainer.epoch if hasattr(trainer, 'epoch') else 0
    total_epochs = trainer.args.epochs if hasattr(trainer, 'args') and hasattr(trainer.args, 'epochs') else 0
    current_batch = trainer.batch if hasattr(trainer, 'batch') else 0
    total_batches = trainer.num_batches if hasattr(trainer, 'num_batches') else 1

    # Update the status
    training_status["current_epoch"] = current_epoch + 1  # Epochs are 0-indexed
    training_status["total_epochs"] = total_epochs

    # Calculate and update the overall training progress (percentage of all epochs)
    overall_progress = int(((current_epoch + (current_batch / total_batches)) / total_epochs) * 100)
    training_status["overall_progress"] = overall_progress

    # Calculate and update the current epoch's progress (percentage of batches in the current epoch)
    epoch_progress = int((current_batch / total_batches) * 100)
    print(f"Epoch: {current_epoch + 1}/{total_epochs}, Batch: {current_batch}/{total_batches}, Progress: {epoch_progress}%")
    training_status["epoch_progress"] = epoch_progress

def train_model(name,organizationId, datasetId, datasetName, modelId, modelType, modelPath):
    global training_status
    training_status["status"] = "running"
    training_status["overall_progress"] = 0
    training_status["epoch_progress"] = 0

    # model = YOLO("yolov8n.pt")
    model = YOLO(modelPath)

    # Add the custom callback to the model
    model.add_callback("on_train_batch_end", progress_callback)
    new_model_path = f"./result/{organizationId}/{datasetName}_{modelType}_{modelId}/weights/best.pt"
    destination_path = f"{"../backend/" if os.getenv('STORAGE_PATH')=='' else os.getenv('STORAGE_PATH')}public/models/{organizationId}/{datasetName}_{modelType}_{modelId}.pt"
    print(destination_path)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    # db.models.delete_many({"isPreTrained":False,"dataset":ObjectId(datasetId),"baseModel":ObjectId(modelId),"organization": ObjectId(organizationId),"type":modelType})
    newModel = db.models.insert_one({ "name": name,"status":"training","isPreTrained":False,"dataset":ObjectId(datasetId),"baseModel":ObjectId(modelId),"organization": ObjectId(organizationId),"type":modelType,"isActive":True})
    model.train(data="./data.yaml" if modelType!="classification" else "dataset" , epochs=10, imgsz=640, project="result/"+organizationId, name=f"{datasetName}_{modelType}_{modelId}", exist_ok=True)
    shutil.rmtree(f"./dataset", ignore_errors=True)
    shutil.copy2(new_model_path, os.path.join(os.path.dirname(destination_path),f"{datasetName}_{modelType}_{modelId}.pt"))
    db.models.update_one({"_id":newModel.inserted_id},{"$set":{"status":"completed","path":f"public/models/{organizationId}/{datasetName}_{modelType}_{modelId}.pt"}})
    # Once training completes, update status
    training_status["status"] = "idle"
    training_status["overall_progress"] = 0
    training_status["epoch_progress"] = 0
    training_status["total_epochs"] = 0
    training_status["current_epoch"] = 0

    

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

# shutil.copy2("./result/66be16adf4fa57825b4cbd20/fishes_objectdetection_6703c8f627de77f27c257980/weights/best.pt","./result/66be16adf4fa57825b4cbd20/fishes_objectdetection_6703c8f627de77f27c257980/weights/best2.pt")

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
