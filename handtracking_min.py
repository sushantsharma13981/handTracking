# It contains min code for tracking
import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0) 

#formality before using this model
mpHands=mp.solutions.hands 
hands=mpHands.Hands()

# drawing lines btw points
mpDraw=mp.solutions.drawing_utils 

pTime, cTime=0,0

while True:
    success, img=cap.read()
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #providing the values
    #print(results.multi_hand_landmarks, mpHands.HAND_CONNECTIONS) 
    
    
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: #handLMS==hand landmarks
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h,w,c=img.shape
                cx, cy=int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                
                # for tracking a specific landmark
                # if id==1:
                #     cv2.circle(img, (cx,cy), 15, (255,0,0), cv2.FILLED)

            #img+landmark_coordinates(points)+connection btw points
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    #To find FPS
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, str(int(fps)), (10,50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2, (255,255,0),1)

    cv2.imshow("image", img)
    cv2.waitKey(1)