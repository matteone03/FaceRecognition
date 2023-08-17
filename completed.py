import cv2
import os

# loads face classifier, you have to insert "face.xml" path on the quotes
cascade_classifier = cv2.CascadeClassifier(r"")

# Open video webcam, may have to change parameter on your pc
cap = cv2.VideoCapture(-1)

while True:
    # read frame from video
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect face
    detections = cascade_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in detections:
        # draw rectangle on the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "face", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('frame', frame)
    
    # close when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

# Release window and all resources
cap.release()
cv2.destroyAllWindows()
