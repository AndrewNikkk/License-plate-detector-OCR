import cv2
import ultralytics
from paddleocr import PaddleOCR


model = ultralytics.YOLO('best.pt')
ocr = PaddleOCR(lang='en')


cap = cv2.VideoCapture(0)


def preprocess(image, scale_factor=4, apply_binarization=True):
    """
    Улучшает изображение для OCR:
    1. Увеличение масштаба
    2. Улучшение контраста (CLAHE в LAB-пространстве)
    3. Опциональная бинаризация (Otsu)
    
    Параметры:
        image: входное изображение (BGR)
        scale_factor: коэффициент увеличения
        apply_binarization: применять ли бинаризацию
        
    Возвращает:
        Обработанное изображение (RGB)
    """
    scaled = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, 
                       interpolation=cv2.INTER_CUBIC)
    

    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    limg = cv2.merge((clahe.apply(l), a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    

    if apply_binarization:
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    else:
        result = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    
    return result


while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame, stream=True)


    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            plate = frame[y1:y2, x1:x2]

            if plate.size != 0:
                plate_rgb = preprocess(plate)
                cv2.imshow("Processed Plate", plate_rgb)
                plate2text = ocr.ocr(plate_rgb, cls = False)
                if plate2text and plate2text[0]:
                    plate2text = plate2text[0][0][1][0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if plate2text is not None:
                print(f"ПЕРЕВОД НОМЕРА В ТЕКСТ:{plate2text}")
            
    cv2.imshow("YOLO Real-Time Detection", frame)
   

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
