# DeepSea Flask API

The **DeepSea Flask API** is a RESTful application designed for managing, training, and deploying deep learning models using the YOLO framework. It supports functionalities like training, real-time progress tracking, stopping training, and performing inference for object detection, segmentation, and classification tasks.

---

## Features

- Train deep learning models using YOLO.
- Real-time training status monitoring.
- Stop ongoing training processes.
- Perform inference on uploaded images for:
  - Object Detection
  - Segmentation
  - Classification.
- Automatically manage model storage and MongoDB integration.

---

## Prerequisites

- Python 3.8 or higher.
- MongoDB with replica set enabled.
- Optional: Network-attached storage (NAS) for model storage.

---

## Setup and Installation

### Step 1: Move to the Project Directory

Navigate to the directory where your project files are located:

```bash
cd /path/to/project-directory
```

### Step 2: Create a Virtual Environment

Create a Python virtual environment to isolate project dependencies:

```bash
python -m venv .
```

### Step 3: Activate the Virtual Environment

Activate the virtual environment to ensure the installed dependencies are used exclusively for this project:

- **On Windows:**

  ```bash
  .\Scripts\activate
  ```

- **On Linux:**
  ```bash
  source ./bin/activate
  ```

### Step 4: Install Dependencies

Install all the required Python libraries listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables

Set up a `.env` file in the project root directory with the following variables:

```dotenv
MONGO_URL=mongodb://localhost:27017/deepsea?replicaSet=myReplicaSet0
STORAGE_PATH='/path/to/storage'
```

### Step 6: Run the Server

Start the Flask development server to serve the API:

```bash
flask run
```

#### Notes:

- **Default URL:** The server will start on `http://127.0.0.1:5000`. Use this URL to access the API.
- **Listen on All Interfaces:** If you want the server to be accessible on all network interfaces, use the following command:
  ```bash
  flask run --host=0.0.0.0
  ```
