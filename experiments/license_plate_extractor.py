import cv2
from ultralytics import YOLO
import easyocr


# Setting up paths to the required files
IMAGE_PATH = 'images/test.jpg'
VIDEO_PATH = 'videos/cars.mp4'
WEIGHTS_PATH = 'yolo-weights/license_plate_detector.pt'

# Load in the image and video
image = cv2.imread(IMAGE_PATH)
# video = cv2.VideoCapture(VIDEO_PATH)


# Initialize the model
model = YOLO(WEIGHTS_PATH)

# Setup the reader for extracting text from the license plate
reader = easyocr.Reader(['en'], model_storage_directory="models/")

# Test the model on an test image
results = model(image)

# Get the coords of the results
for result in results:
    bboxes = result.boxes
    for bbox in bboxes:
        # print(bbox.xyxy)
        x1, y1, x2, y2 = bbox.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        license_plate = image[y1:y2, x1:x2]
        ocr_results = reader.readtext(license_plate)

        for (_, text, _) in ocr_results:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))



cv2.imshow("image", image)
# cv2.imshow("license-plate", license_plate)
cv2.waitKey(0)