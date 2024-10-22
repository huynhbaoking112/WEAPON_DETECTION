from ultralytics import YOLO
import cvzone
import cv2
import math

# Mở video từ file
cap = cv2.VideoCapture('gun10.mp4')
# cap = cv2.VideoCapture(0)
model = YOLO('weapon.pt')

# Đọc các lớp
classnames = [
    'axe', 'bomb', 'bow', 'cleaver', 'cutlass',
    'katana', 'knife', 'mace', 'machine gun',
    'morningstar', 'pistol', 'rifle', 'rocket launcher',
    'scabbard', 'scope', 'shield', 'shotgun',
    'sickle', 'smg', 'sniper rifle', 'spear',
    'sword', 'war hammer'
]

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Dừng nếu không còn frame nào

    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    # Lấy thông tin bbox, confidence và tên lớp
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 30],
                                   scale=1.5, thickness=2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Nhấn 'q' để thoát

cap.release()
cv2.destroyAllWindows()
