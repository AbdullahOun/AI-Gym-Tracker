import cv2
import mediapipe as mp
import numpy as np
import csv

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture('./Up.mp4')
class_name = None

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        results = holistic.process(image)
        
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            if class_name:  # Only write to file if class_name is set
                row = pose_row
                row.insert(0, class_name)
                
                with open('coords.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
                
        except:
            pass
        
        window_name = 'Webcam Feed'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_rect = cv2.getWindowImageRect(window_name)
        
        cv2.resizeWindow(window_name, 500, 640)
        cv2.imshow(window_name, image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u'):
            class_name = 'up'
        elif key == ord('D'):
            class_name = 'down'
        else:
            class_name = None  # Reset class_name if no relevant key is pressed

cap.release()
cv2.destroyAllWindows()
