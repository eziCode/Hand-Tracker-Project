import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

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

def hand_landmark_callback(result, image, timestamp_ms):
    print(f"Results received at {timestamp_ms} ms")
    if result.hand_landmarks:
        print(result.hand_landmarks)

def show_camera_feed(capture: cv2.VideoCapture, capture_type: str):
    model_path = 'C:\\Users\\ezraa\\Downloads\\hand_landmarker.task'
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=hand_landmark_callback
    )

    with HandLandmarker.create_from_options(options) as hand_landmarker:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            if capture_type == "vertical":
                draw_coordinate_lines_for_vertical_capture(frame)
            elif capture_type == "horizontal":
                draw_coordinate_lines_for_horizontal_capture(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            hand_landmarker.detect_async(mp_image, timestamp_ms=int(time.time() * 1000))

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
