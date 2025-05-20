# It contains min code for tracking
import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2,model_complexity=1,
                  detectionCon=0.5, trackCon=0.5):
        self.mode=mode
        self.maxhands=maxHands
        self.model_complexity=model_complexity
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpHands=mp.solutions.hands 
        self.hands=self.mpHands.Hands(self.mode, self.maxhands,
                                      self.model_complexity,
                                        self.detectionCon,
                                        self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils 
    

    def findHands(self, img, draw=True):
        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks: 
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, 
                                           self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList=[]

        if self.results.multi_hand_landmarks:
            #for specific hand
            myHand=self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx, cy=int(lm.x*w), int(lm.y*h)
              
                lmList.append([id,cx,cy])
                if draw==0:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

        return lmList
    

def main():
    pTime, cTime=0,0
    cap=cv2.VideoCapture(0) 
    detector=handDetector()
    while True:
        success, img=cap.read()
        img=detector.findHands(img)
        lmlist= detector.findPosition(img)
        if len(lmlist)!=0:
            print(lmlist[0])

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img, str(int(fps)), (10,50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2, (255,255,0),1)

        cv2.imshow("image", img)
        cv2.waitKey(1)


#dummy code in main for working of module
if __name__=="__main__" :
    main()
    