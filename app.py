import cv2
import ultralytics


model = ultralytics.YOLO('best.pt')


cap = cv2.VideoCapture(0)


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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    cv2.imshow("YOLO Real-Time Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()