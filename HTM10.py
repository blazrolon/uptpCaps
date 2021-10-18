import mediapipe as mp
import numpy as np
import cv2
import os
from threading import Timer
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#Width and height of the picture
w = 640
h = 480

#Path to the folder where the pictures are saved
path = "C:\\testAI\\caps\\pictures"

#Flags
operate = True
has_5s_left = False

TIMER = 3

def get_coordinates(index, hand, results):
    output = np.empty((4,2))

    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            
            # Extract Coordinates
            if label == "Left":
                cL4 = tuple(np.multiply(
                    np.array((hand.landmark[mp_hands.HandLandmark.THUMB_MCP].x, hand.landmark[mp_hands.HandLandmark.THUMB_MCP].y)),
                [w, h]).astype(int))
                output[0] = cL4

                cL8 = tuple(np.multiply(
                    np.array((hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x, hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y)),
                [w, h]).astype(int))
                output[1] = cL8

            elif label == "Right":
                cR4 = tuple(np.multiply(
                    np.array((hand.landmark[mp_hands.HandLandmark.THUMB_MCP].x, hand.landmark[mp_hands.HandLandmark.THUMB_MCP].y)),
                [w, h]).astype(int))
                output[2] = cR4

                cR8 = tuple(np.multiply(
                    np.array((hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x, hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y)),
                [w, h]).astype(int))            
                output[3] = cR8

    return output

def take_screenshot():
    prev = time.time()
    global TIMER
    while TIMER >= 0:
        cv2.putText(image, str(TIMER), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cur = time.time()
        if cur-prev >= 1:
            prev = cur
            TIMER = TIMER-1
 
    cv2.imwrite(os.path.join(path, file_name),crop)
    print("Saved")
    global operate 
    operate = False

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands: 
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
                                        mp_drawing.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 4),
                                        mp_drawing.DrawingSpec(color = (250, 44, 250), thickness = 2, circle_radius = 2),
                                        )

                #Get coordinates and draw the frame
                coord = None
                coord = get_coordinates(num, hand, results)
                if coord.all() and len(results.multi_hand_landmarks)>1: 
                    p1 = np.array((coord[1][0], coord[2][1])).astype(int)
                    cv2.circle(image, p1, 5, (255, 255, 255), cv2.FILLED)
                    cv2.putText(image, "p1", p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    p2 = np.array((coord[1][0], coord[0][1])).astype(int)
                    cv2.circle(image, p2, 5, (255, 255, 255), cv2.FILLED)
                    cv2.putText(image, "p2", p2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    p3 = np.array((coord[3][0], coord[0][1])).astype(int)
                    cv2.circle(image, p3, 5, (255, 255, 255), cv2.FILLED)
                    cv2.putText(image, "p3", p3, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    p4 = np.array((coord[3][0], coord[2][1])).astype(int)
                    cv2.circle(image, p4, 5, (255, 255, 255), cv2.FILLED)
                    cv2.putText(image, "p4", p4, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    cv2.line(image, p1, p2, (255, 255, 255), 3)
                    cv2.line(image, p2, p3, (255, 255, 255), 3)
                    cv2.line(image, p3, p4, (255, 255, 255), 3)
                    cv2.line(image, p4, p1, (255, 255, 255), 3)
                    
                    #Save a picture of the frame after 5 seconds of detecting the hands
                    startX = min(p1[0], p3[0]).astype(int) 
                    endX = max(p1[0], p3[0]).astype(int) 
                    startY = min(p1[1], p3[1]).astype(int) 
                    endY = max(p1[1], p3[1]).astype(int) 
                    
                    crop = frame[startY:endY, startX:endX]
                    
                    num_elements = len([item for item in os.listdir(path) 
                        if os.path.isfile(os.path.join(path, item))])
                    file_name = str(num_elements).zfill(5)+'.jpg'

                    if has_5s_left == False:
                        has_5s_left = True
                        save_frame = Timer(TIMER, take_screenshot)
                        save_frame.start()
  
        # Show our image    
        cv2.imshow('Hand Tracking', image)

        if operate == False:
            break
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()