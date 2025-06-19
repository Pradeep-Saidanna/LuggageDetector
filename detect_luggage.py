import cv2
import torch
import pyttsx3
import time

model = torch.hub.load('ultralytics/yolov5','yolov5s')
target_label = ['backpack', 'handbag','suitcase']

engine = pyttsx3.init()
engine.setProperty('rate',160)

cap = cv2.VideoCapture("C:\luggage_detector\detection_video2.mp4")
print("Starting luggage detection... Press 'Q' to quit.")

last_spoken = set()
speak_delay = 1
last_spoken_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret :
        print("Failed to grab frame.")
        break
    frame = cv2.resize(frame, (640, 480))
    results = model(frame)

    detections = results.pandas().xyxy[0]
    current_detections = set()


    for index, row in detections.iterrows():
        label = row['name']
        confidence = row['confidence']

        if label in target_label and confidence > 0.5:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 244, 0), 2)
            current_detections.add(label)

    if time.time() - last_spoken_time > speak_delay :
        for item in current_detections :
            if item not in last_spoken:
                engine.say(f"{item} detected")
        engine.runAndWait()
        last_spoken = current_detections
        last_spoken_time = time.time()
    
    cv2.imshow("Luggage Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Existing...")
        break
cap.release()
cv2.destroyAllWindows()
        

#to run - "python detect_luggage.py"