from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
import base64

app = Flask(__name__, static_folder='.', static_url_path='')

# --------------------------
# Render writable directory setup
# --------------------------
BASE_DIR = '/tmp' if os.environ.get("RENDER") else '.'
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.pkl")
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

    print(f"[INFO] Registering {name} with {len(imgs)} images")

    reps = []
    for i, img_data in enumerate(imgs):
        img = decode_image(img_data)
        img_path = os.path.join(user_dir, f"{i}.jpg")
        cv2.imwrite(img_path, img)
        try:
            rep = DeepFace.represent(
                img_path,  # ✅ use path for better stability
                model_name="Facenet",
                detector_backend="retinaface",  # more accurate than opencv
                enforce_detection=False
            )[0]["embedding"]
            reps.append(rep)
        except Exception as e:
            print(f"[ERROR] Embedding failed for image {i}: {e}")

    if reps:
        embeddings[name] = np.mean(reps, axis=0)
        save_embeddings(embeddings)
        print(f"[OK] {name} registered successfully with {len(reps)} embeddings")
        return jsonify({"status": "ok", "message": f"User '{name}' registered successfully."})
    else:
        print("[FAIL] No valid embeddings generated.")
        return jsonify({"status": "error", "message": "Registration failed: no valid faces detected"})

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
    try:
        detected = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
        if not detected:
            return jsonify({"status": "ok", "bbox": None, "name": "No face", "distance": 0})

        d = detected[0]
        x, y, w, h = d['facial_area'].values()
        emb = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)[0]["embedding"]

        min_dist, matched_name = 999, "Unknown"
        for uname, uemb in embeddings.items():
            dist = np.linalg.norm(uemb - emb)
            if dist < min_dist:
                min_dist = dist
                matched_name = uname

        threshold = 10  # tweak if needed
        if min_dist < threshold:
            print(f"[MATCH] {matched_name} ({min_dist:.3f})")
            return jsonify({"status": "ok", "bbox": [x, y, w, h], "name": matched_name, "distance": float(min_dist),
                             "img_w": frame.shape[1], "img_h": frame.shape[0]})
        else:
            print(f"[NO MATCH] Closest {matched_name} ({min_dist:.3f})")
            return jsonify({"status": "ok", "bbox": [x, y, w, h], "name": "Unknown", "distance": float(min_dist),
                             "img_w": frame.shape[1], "img_h": frame.shape[0]})
    except Exception as e:
        print("[ERROR] Recognition:", e)
        return jsonify({"status": "error", "message": str(e)})
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
    return send_from_directory(os.path.dirname(__file__), "index.html")

# --------------------------
# Run Flask
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

