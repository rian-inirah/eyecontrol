import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Blink detection thresholds
BLINK_EAR_THRESHOLD = 0.2
LONG_BLINK_TIME = 1  # 1 second for long blink

# Variables for blink tracking
blink_start = None
blink_count = 0

# For vertical eye movement tracking
movement_history = []
history_length = 7  # Increased for smoother scrolling

# Scrolling sensitivity settings
SCROLL_THRESHOLD = 0.01  # Increased to prevent over-sensitivity
SCROLL_SPEED_FACTOR = 5000  # Reduced to slow down scrolling

def calculate_EAR(eye_points):
    """Calculate Eye Aspect Ratio (EAR) to detect blinks."""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

def get_eye_landmarks(landmarks, indices):
    """Extract coordinates for given eye landmarks."""
    return np.array([(landmarks[i].x, landmarks[i].y) for i in indices])

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot open webcam!")
    exit()

print("✅ Webcam opened successfully!")

# Define eye landmark indices for both eyes
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("❌ Error: Failed to read frame from camera!")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Get landmarks for both eyes
        left_eye = get_eye_landmarks(face_landmarks.landmark, left_eye_indices)
        right_eye = get_eye_landmarks(face_landmarks.landmark, right_eye_indices)

        # Calculate Eye Aspect Ratio (EAR) for blink detection
        left_EAR = calculate_EAR(left_eye)
        right_EAR = calculate_EAR(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2  # Take average of both eyes

        # Detect blinks
        if avg_EAR < BLINK_EAR_THRESHOLD:
            if blink_start is None:
                blink_start = time.time()
            elif time.time() - blink_start > LONG_BLINK_TIME:
                print("🚨 Long Blink Detected - Cancel Action")
                blink_start = None
                blink_count = 0
        else:
            if blink_start is not None:
                blink_count += 1
                blink_start = None

            if blink_count == 1:
                print("✅ Single Blink - Select Action Triggered")
                blink_count = 0
            elif blink_count == 2:
                print("✅✅ Double Blink - Confirm Action Triggered")
                blink_count = 0

        # Track vertical eye movement for scroll control
        eye_center_y = ((left_eye[1][1] + left_eye[5][1]) + (right_eye[1][1] + right_eye[5][1])) / 4

        # Store recent movements for smooth scrolling
        movement_history.append(eye_center_y)

        if len(movement_history) > history_length:
            movement_history.pop(0)  # Keep only last few values

        if len(movement_history) == history_length:
            avg_movement = movement_history[-1] - movement_history[0]  # Compare oldest & newest value

            # Dead zone to prevent unwanted small movements
            if abs(avg_movement) > SCROLL_THRESHOLD:
                scroll_speed = int(abs(avg_movement) * SCROLL_SPEED_FACTOR)

                if avg_movement > 0:
                    pyautogui.scroll(-scroll_speed)  # Scroll down
                    print(f"⬇️ Scrolling Down with speed {scroll_speed}")
                else:
                    pyautogui.scroll(scroll_speed)  # Scroll up
                    print(f"⬆️ Scrolling Up with speed {scroll_speed}")

        # Draw eye landmarks for visualization
        for eye in [left_eye, right_eye]:
            for (x, y) in eye:
                x, y = int(x * frame.shape[1]), int(y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    else:
        print("🚫 No face detected")

    # Always show the frame (even if no face detected)
    cv2.imshow("Eye Control Interface", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
