from ultralytics import YOLO
import cv2 as cv2
import cvzone
import math
import supervision as sv
import numpy as np

# Load in the pretrained model
pretrained_model = YOLO('yolo-weights/best.pt')


# Setup video capture
VIDEO_PATH = 'videos/cars.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(3, 1280)
cap.set(4, 720)

# Setup class name
class_name = "license-plate"

# Setting up mask
mask = cv2.imread('images/mask.png')

# Setting up endpoints of the line
coords = [(300, 491), (1236, 491)]

# Tracking objects in the video
byte_tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()

# define tracking fucntions
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        results = pretrained_model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detectiosn = byte_tracker.update_with_detections(detections)
        labels = [
            f"#{tracker_id} {class_name} {confidence:0.2f}"
            for _, _, confidence, _, tracker_id in detections
            ]
        return annotator.annotate(frame.copy(), detections, labels)

# set up car count for tracking
total_count = []

# Sending every model in the video to the model for license plate detection
while True:
    # Reading in frame by frame
    success, img = cap.read()

    # Apply mask to reduce computation and for best performance
    img_region = cv2.bitwise_and(img, mask)

    # Sending every frame to the model for license plate detection
    results = pretrained_model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # this line is commented because we will display the boxes from result tracker
            # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255))

            # for finding the confidence
            conf = math.ceil(box.conf[0] * 100) / 100


            # for class name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f"{class_name} {conf} ", (max(0, x1), max(35, y1)),
                                scale=1, thickness=1)

    # display the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)