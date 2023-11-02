from ultralytics import YOLO
import cv2 as cv2
import easyocr
import math
from utils.sort import *
import numpy as np

# Load in the pretrained model
pretrained_model = YOLO('yolo-weights/license_plate_detector.pt')


# Setup video capture
VIDEO_PATH = 'videos/cars.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(3, 1280)
cap.set(4, 720)


# Setup mask for better performance
mask = cv2.imread('images/mask.png')

# Setup class name
class_name = "license-plate"

# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Setup reader for ocr
reader = easyocr.Reader(['en'], model_storage_directory='models/')

# Sending every model in the video to the model for license plate detection
while True:
    # Reading in frame by frame
    success, img = cap.read()

    # pass the image through the mask
    img_region = cv2.bitwise_and(img, mask)

    # Sending every frame to the model for license plate detection
    results = pretrained_model(img, stream=True)

    # Setup an empty detections array
    detections = np.empty([0, 5])

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
            # cvzone.putTextRect(img, f"{class_name} {conf} ", (max(0, x1), max(35, y1)),
            #                     scale=1, thickness=1)

            # Adding the detections to the empty array
            current_array = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, current_array))

    tracker_results = tracker.update(detections)
    for result in tracker_results:
            x1, y1, x2, y2, obj_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            license_plate = img[y1:y2, x1:x2]
            ocr_results = reader.readtext(license_plate)

            for (_, text, _) in ocr_results:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
                cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))


    # display the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

