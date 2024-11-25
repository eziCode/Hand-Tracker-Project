import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score


previous_landmarks = None
y_values_to_z_values = []

def plot_y_vs_z():
    global y_values_to_z_values
    y_vals = np.array([val[0] for val in y_values_to_z_values])
    z_vals = np.array([val[1] for val in y_values_to_z_values])

    linear_model = LinearRegression()
    y_vals_reshaped = y_vals.reshape(-1, 1)
    linear_model.fit(y_vals_reshaped, z_vals)
    linear_predictions = linear_model.predict(y_vals_reshaped)
    linear_r2 = r2_score(z_vals, linear_predictions)

    # Fit a quadratic regression
    quadratic_features = np.column_stack([y_vals, y_vals**2])
    quadratic_model = LinearRegression()
    quadratic_model.fit(quadratic_features, z_vals)
    quadratic_predictions = quadratic_model.predict(quadratic_features)
    quadratic_r2 = r2_score(z_vals, quadratic_predictions)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_vals, z_vals, color='blue', label='Data Points')

    plt.plot(y_vals, linear_predictions, color='red', label=f'Linear Fit ($R^2 = {linear_r2:.2f}$)')

    sorted_indices = np.argsort(y_vals)
    plt.plot(
        y_vals[sorted_indices], 
        quadratic_predictions[sorted_indices], 
        color='green', 
        label=f'Quadratic Fit ($R^2 = {quadratic_r2:.2f}$)'
    )

    plt.title('Correlation between Y and Z Values with Regression Fits')
    plt.xlabel('Y Value')
    plt.ylabel('Z Value')
    plt.legend()
    plt.grid(True)
    plt.show()

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

def hand_landmark_callback(result, __, ___):
    global previous_landmarks
    if result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name
            handedness_score = result.handedness[i][0].score
            # for j, landmark in enumerate(hand_landmarks):
            #     print(f"Landmark {j}: x={landmark.x}, y={landmark.y}, z={landmark.z}")
            previous_landmarks = (hand_landmarks, handedness, handedness_score)

            # draw_landmarks(hand_landmarks, handedness, handedness_score, frame, width, height)
    else:
        previous_landmarks = None

def draw_landmarks_for_vertical_capture(landmarks, handedness, handedness_score, frame):
    height, width = frame.shape[:2]
    x, y = landmarks[8].x, landmarks[8].y
    frame_relative_position = (x * width, y * height)
    print(frame_relative_position)
    draw_landmarks(landmarks, handedness, handedness_score, frame, width, height)

def draw_landmarks_for_horizontal_capture(landmarks, handedness, handedness_score, frame):
    global y_values_to_z_values
    height, width = frame.shape[:2]
    average_y_value_of_hand = sum(landmark.y for landmark in landmarks if landmark != landmarks[8]) / (len(landmarks) - 1)
    # average_z_value_of_hand = abs(sum(landmark.z for landmark in landmarks if landmark != landmarks[8]) / (len(landmarks) - 1))
    # y_values_to_z_values.append((average_y_value_of_hand, average_z_value_of_hand))
    average_y_pixel = int(average_y_value_of_hand * height)
    cv2.line(frame, (0, average_y_pixel), (width, average_y_pixel), (0, 255, 0), 2)
    draw_landmarks(landmarks, handedness, handedness_score, frame, width, height, average_y_pixel)

def draw_landmarks(landmarks, handedness, handedness_score, frame, width, height, average_y_pixel=float('inf')):
    height, width = frame.shape[:2]
    for i, landmark in enumerate(landmarks):
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        if i == 8 and landmark.y > (average_y_pixel / height):
            print("point clicked")
            print(f"Landmark {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(frame, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    cv2.putText(frame, f"{handedness} ({handedness_score:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def show_camera_feed(capture: cv2.VideoCapture, capture_type: str):
    # model_path = 'C:\\Users\\ezraa\\Downloads\\hand_landmarker.task'
    model_path = '/Users/ezraakresh/Downloads/hand_landmarker.task'
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
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            if previous_landmarks and capture_type == "horizontal":
                draw_landmarks_for_horizontal_capture(previous_landmarks[0], previous_landmarks[1], previous_landmarks[2], frame)
            elif previous_landmarks and capture_type == "vertical":
                draw_landmarks_for_vertical_capture(previous_landmarks[0], previous_landmarks[1], previous_landmarks[2], frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            hand_landmarker.detect_async(mp_image, timestamp_ms=int(time.time() * 1000))

            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # plot_y_vs_z()

    capture.release()
    cv2.destroyAllWindows()

def error_message(message: str):
    print(f"Error: {message}")
    exit()

vertical_capture = cv2.VideoCapture(0)
# horizontal_capture = cv2.VideoCapture(0)


# if not vertical_capture.isOpened() or not horizontal_capture.isOpened():
#     error_message("Failed to open camera")

# show_camera_feed(horizontal_capture, "horizontal")
show_camera_feed(vertical_capture, "vertical")

# point 8 is the tip of the index finger
# take average y coordinate of points in hand and then compare to y coordinate of point 8

# show_camera_feed(vertical_capture, "vertical")
