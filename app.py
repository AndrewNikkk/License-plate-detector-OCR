import cv2
import ultralytics
import numpy as np
import re
from paddleocr import PaddleOCR
from google.colab.patches import cv2_imshow

from preprocess import enhance_text_ocr

model = ultralytics.YOLO('best.pt')
ocr = PaddleOCR(lang='en')

cap = cv2.VideoCapture('video_yolo_test2.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('processed_output1.mp4', fourcc, fps, (frame_w, frame_h))

with open('correct_license_plates.txt', 'w') as file:

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                plate = frame[y1:y2, x1:x2]

                plate2text = None
                if plate.size != 0:
                    plate_rgb = enhance_text_ocr(plate)
                    plate2text_raw = ocr.ocr(plate_rgb, cls=False)
                    if plate2text_raw and plate2text_raw[0]:
                        plate2text = plate2text_raw[0][0][1][0]
                        plate2text = plate2text.replace(" ", "").upper()

                        pattern = r'^[A-Z]\d{6}$' \
                                  r'|^[A-Z]{2}\d{5,6}$' \
                                  r'|^\d{3}[A-Z]{1,2}\d{3,5}$' \
                                  r'|^\d{4}[A-Z]{2}\d{2}$' \
                                  r'|^[A-Z]{2}\d{6}$'

                        if re.fullmatch(pattern, plate2text):
                            print(f"✅ Корректный номер: {plate2text}")
                            
                            file.write(plate2text + '\n') # сохранение номеров в файл
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, plate2text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            print(f"❌ Отклонённый номер: {plate2text}")
                            plate2text = None

        out.write(frame)

        if frame_count % 30 == 0:
            cv2_imshow(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
