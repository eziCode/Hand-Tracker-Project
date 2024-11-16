import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_coordinate_lines_for_vertical_capture(frame):
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    cv2.line(frame, (0, center_y), (width, center_y), (0, 255, 0), 2)
    cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 2)


def draw_coordinate_lines_for_horizontal_capture(frame):
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    cv2.line(frame, (0, center_y + 95), (width, center_y + 105), (0, 255, 0), 2)
    cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 2)


def show_camera_feed(capture: cv2.VideoCapture, capture_type: str):
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        # if capture_type == "vertical":
        #     draw_coordinate_lines_for_vertical_capture(frame)
        # elif capture_type == "horizontal":
        #     draw_coordinate_lines_for_horizontal_capture(frame)

        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

def error_message(message: str):
    print(f"Error: {message}")
    exit()

vertical_capture = cv2.VideoCapture(1)
horizontal_capture = cv2.VideoCapture(0)

if not vertical_capture.isOpened() or not horizontal_capture.isOpened():
    error_message("Failed to open camera")

# show_camera_feed(horizontal_capture, "horizontal")
show_camera_feed(vertical_capture, "vertical")
