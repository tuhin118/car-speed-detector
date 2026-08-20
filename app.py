from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# YOLO model
model = YOLO("yolo11n.pt")

# Vehicle classes in COCO
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]

    if video.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = video.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    video.save(filepath)

    return jsonify({
        "status": "success",
        "filename": filename,
        "message": "Video uploaded successfully"
    })


@app.route("/status")
def status():
    return jsonify({
        "detector": "YOLO",
        "vehicles": list(VEHICLE_CLASSES.values()),
        "camera": "browser camera supported",
        "speed_detection": "estimated"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
