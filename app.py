from flask import Flask, request, jsonify, render_template_string, send_from_directory
import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
import base64

app = Flask(__name__, static_folder='.', static_url_path='')

DATASET_DIR = "dataset"
EMBEDDINGS_FILE = "embeddings.pkl"
os.makedirs(DATASET_DIR, exist_ok=True)

# --------------------------
# Load / Save embeddings
# --------------------------
def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_embeddings(embeddings):
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)

embeddings = load_embeddings()

# --------------------------
# Helper - Convert base64 to image
# --------------------------
def decode_image(img_data):
    img_data = img_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# --------------------------
# Register route
# --------------------------
@app.route("/register_images", methods=["POST"])
def register_images():
    data = request.json
    name = data.get("name")
    imgs = data.get("images", [])
    if not name or not imgs:
        return jsonify({"status": "error", "message": "Invalid data"})

    user_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    for i, img_data in enumerate(imgs):
        img = decode_image(img_data)
        cv2.imwrite(os.path.join(user_dir, f"{i}.jpg"), img)

    # Compute embeddings
    reps = []
    for img_data in imgs:
        img = decode_image(img_data)
        try:
            emb = DeepFace.represent(img, model_name='Facenet', enforce_detection=False)[0]["embedding"]
            reps.append(emb)
        except Exception as e:
            print("Embedding error:", e)
    if reps:
        embeddings[name] = np.mean(reps, axis=0)
        save_embeddings(embeddings)
        return jsonify({"status": "ok", "message": f"User '{name}' registered successfully."})
    else:
        return jsonify({"status": "error", "message": "No face detected"})

# --------------------------
# Recognize route
# --------------------------
@app.route("/recognize_image", methods=["POST"])
def recognize_image():
    data = request.json
    img_data = data.get("image")
    if not img_data:
        return jsonify({"status": "error", "message": "No image data"})

    frame = decode_image(img_data)
    bbox, name, dist = None, "Unknown", 0.0

    try:
        detections = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
        if detections:
            d = detections[0]
            x, y, w, h = d['facial_area'].values()
            bbox = [x, y, w, h]
            emb = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False)[0]["embedding"]

            # Compare with stored embeddings
            min_dist, matched_name = 999, "Unknown"
            for uname, uemb in embeddings.items():
                dist = np.linalg.norm(uemb - emb)
                if dist < min_dist:
                    min_dist = dist
                    matched_name = uname
            if min_dist < 10:  # adjustable threshold
                name = matched_name
                dist = min_dist
    except Exception as e:
        print("Recognition error:", e)

    h, w = frame.shape[:2]
    return jsonify({"status": "ok", "bbox": bbox, "name": name, "distance": dist, "img_w": w, "img_h": h})

# --------------------------
# Delete User
# --------------------------
@app.route("/delete", methods=["POST"])
def delete_user():
    data = request.json
    name = data.get("name")
    if name in embeddings:
        embeddings.pop(name)
        save_embeddings(embeddings)
    user_dir = os.path.join(DATASET_DIR, name)
    if os.path.exists(user_dir):
        for f in os.listdir(user_dir):
            os.remove(os.path.join(user_dir, f))
        os.rmdir(user_dir)
    return jsonify({"message": f"Deleted {name} successfully"})

# --------------------------
# List Users
# --------------------------
@app.route("/users")
def list_users():
    return jsonify({"users": list(embeddings.keys())})

# --------------------------
# Serve index.html from same folder
# --------------------------
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# --------------------------
# Run Flask
# --------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    # debug=False for production
    app.run(host="0.0.0.0", port=port, debug=False)
