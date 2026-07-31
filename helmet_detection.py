import os
import cv2
import tempfile

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

model = YOLO(MODEL_PATH)

def detect_helmet(image):

    results = model(image)

    result = results[0]

    detected_image = result.plot()


    helmet_count = 0
    no_helmet_count = 0
    confidence_scores = []


    for box in result.boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        confidence_scores.append(confidence)


        class_name = model.names[class_id]


        if class_name == "With Helmet":
            helmet_count += 1

        elif class_name == "Without Helmet":
            no_helmet_count += 1



    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

    else:
        avg_confidence = 0



    return (
        detected_image,
        helmet_count,
        no_helmet_count,
        avg_confidence
    )

# -------- VIDEO DETECTION --------

def detect_video(video_path):

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    output_path = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ).name


    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        results = model(frame)

        detected_frame = results[0].plot()

        writer.write(detected_frame)


    cap.release()
    writer.release()


    return output_path
# -------- LIVE WEBCAM DETECTION --------

def detect_webcam(frame):

    results = model(frame)

    detected_frame = results[0].plot()

    return detected_frame