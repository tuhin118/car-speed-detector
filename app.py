from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

model = YOLO("yolo11n.pt")

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

    filename = f"{uuid.uuid4().hex}.mp4"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    video.save(filepath)

    cap = cv2.VideoCapture(filepath)

    if not cap.isOpened():
        return jsonify({"error": "Could not open video"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30

    frame_count = 0
    detected = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # Process every 3rd frame
        if frame_count % 3 != 0:
            continue

        # YOLO vehicle tracking
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        for result in results:
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id not in VEHICLE_CLASSES:
                    continue

                if confidence < 0.40:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Tracking ID
                track_id = None

                if box.id is not None:
                    track_id = int(box.id[0])

                detected.append({
                    "id": track_id,
                    "type": VEHICLE_CLASSES[class_id],
                    "confidence": round(confidence, 2),
                    "x": (x1 + x2) // 2,
                    "y": (y1 + y2) // 2,
                    "frame": frame_count,
                    "time": round(frame_count / fps, 3)
                })

    cap.release()

    try:
        os.remove(filepath)
    except OSError:
        pass

    return jsonify({
        "status": "success",
        "fps": fps,
        "frames": frame_count,
        "vehicles": detected,
        "message": "Vehicle detection and tracking completed"
    })


@app.route("/status")
def status():
    return jsonify({
        "detector": "YOLO",
        "tracker": "ByteTrack",
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
