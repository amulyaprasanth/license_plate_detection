import cv2

def get_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(f"Mouse clicked at ({x}, {y})")

cap = cv2.VideoCapture('../videos/cars.mp4')  # Replace with your video file

cv2.namedWindow('Video with Line')
cv2.setMouseCallback('Video with Line', get_mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video with Line', frame)

    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
