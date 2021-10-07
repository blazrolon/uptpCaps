import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

import threading 
from threading import Timer
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

w = 640
h = 480

def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [w, h]).astype(int))
            
            output = text, coords
            
    return output

def get_coord(index, hand, results):
    output = None
    
    output=np.empty((4,2))

    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            
            # Extract Coordinates
            if label == "Left":
                cL4 = tuple(np.multiply(
                    np.array((hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y)),
                [w, h]).astype(int))
                output[0]=cL4
                
                cL8 = tuple(np.multiply(
                    np.array((hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)),
                [w, h]).astype(int))
                output[1]=cL8
                
            elif label == "Right":
                cR4 = tuple(np.multiply(
                    np.array((hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y)),
                [w, h]).astype(int))
                output[2]=cR4

                cR8 = tuple(np.multiply(
                    np.array((hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)),
                [w, h]).astype(int))            
                output[3]=cR8
            
    return output

def take_screenshot(file_name,crop):
    #print(crop)
    global flag 
    flag = False
    #time.sleep(2)
    cv2.imwrite(os.path.join("C:\\testAI\\caps\\pictures2", file_name),crop)
    print("Saved")



flag = True

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
        #print(results)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                
                # Render left or right detection
                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                coord = get_coord(num, hand, results)
                if coord.all() and len(results.multi_hand_landmarks)>1:
                    p1 = np.array((coord[0][0],coord[3][1])).astype(int)
                    cv2.circle(image, p1, 5, (0, 255, 255), cv2.FILLED)
                    
                    p2 = np.array((coord[0][0],coord[1][1])).astype(int)
                    cv2.circle(image, p2, 5, (0, 0, 255), cv2.FILLED)

                    p3 = np.array((coord[2][0],coord[1][1])).astype(int)
                    cv2.circle(image, p3, 5, (255, 255, 0), cv2.FILLED)
                    
                    p4 = np.array((coord[2][0],coord[3][1])).astype(int)
                    cv2.circle(image, p4, 5, (255, 0, 0), cv2.FILLED)

                    cv2.line(image, p1, p2, (255, 255, 255), 3)
                    cv2.line(image, p2, p3, (255, 255, 255), 3)
                    cv2.line(image, p3, p4, (255, 255, 255), 3)
                    cv2.line(image, p4, p1, (255, 255, 255), 3)

                    startX = p1[0].astype(int) 
                    endX = p3[0].astype(int) 
                    startY = p1[1].astype(int)
                    endY = p3[1].astype(int)

                    num_elements = len([item for item in os.listdir("C:\\testAI\\caps\\pictures2") if os.path.isfile(os.path.join("C:\\testAI\\caps\\pictures2", item))])
                    file_name = str(num_elements).zfill(5)+'.jpg'
                    
                    crop = frame[startY:endY,startX:endX]
                    
                    snap = Timer(5, take_screenshot, [file_name,crop])
                    snap.start()

                    # t1 = threading.Thread(target=snap1(p1, p2, p3, p4))  
                    # t2 = threading.Thread(target=snap2(p1,p2,p3,p4))  
                    # t1.start()
                    # t2.start()
                    if flag == False:
                        break
        # Save our image    
        #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Tracking', image)

        if flag == False:
            break
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()