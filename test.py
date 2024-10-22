from ultralytics import YOLO
# import cvzone
import cv2
import math

# Đọc hình ảnh
image_path = 'gun3.jpg'  # Thay 'path_to_your_image.jpg' bằng đường dẫn đến hình ảnh của bạn
frame = cv2.imread(image_path)
frame = cv2.resize(frame, (640, 640 ))

# Tải mô hình
model = YOLO('weapon.pt')

# Thực hiện nhận diện
result = model(frame)

# Lấy thông tin bounding box, confidence và tên lớp để xử lý
for info in result:
    boxes = info.boxes
    for box in boxes:
        confidence = box.conf[0]
        confidence = math.ceil(confidence * 100)
        Class = int(box.cls[0])
        
        if confidence > 50:  # Chỉ xử lý các kết quả có độ tin cậy lớn hơn 50%
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Vẽ hình chữ nhật quanh đối tượng
            # cvzone.putTextRect(frame, f'Class: {Class} {confidence}%', [x1 + 8, y1 + 30],
                            #    scale=1.5, thickness=2)  # Thêm văn bản vào hình ảnh

# Hiển thị hình ảnh với kết quả nhận diện
cv2.imshow('Detected Image', frame)

# Nhấn phím bất kỳ để thoát
cv2.waitKey(0)
cv2.destroyAllWindows()
