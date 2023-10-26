# Import the required packages
from easyocr import Reader
import cv2

# load the image and resize it
image = cv2.imread('images/test.jpg')
image = cv2.resize(image, (800, 600))

# convert the input image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blur, 10, 200)
cv2.imshow('Canny', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()