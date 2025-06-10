import cv2

cap = cv2.VideoCapture(3)

if not cap.isOpened():
    print("❌ Cannot open camera. Try changing the index to 1, 2, 3...")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow('Camera Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
