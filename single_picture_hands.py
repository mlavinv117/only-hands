import cv2
import mediapipe as mp
import os

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5,):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
            if draw:
                cv2.circle(image,(cx,cy), 5 , (255,0,255), cv2.FILLED)

        return lmlist

tracker = handTracker()

path = '/home/lvizcaino/code/ManuelLavinSLB/only-hands/data/input_picture.png'
if os.path.exists(path):
    image = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
else:
    print("Path does not exist:", path)

image = tracker.handsFinder(image)
lmList = tracker.positionFinder(image)
if len(lmList) != 0:
    max_width = 0
    min_width = 1000000
    max_height = 0
    min_height = 1000000
    for point in lmList:
        if point[1]>max_width:
            max_width=point[1]
        if point[1]<min_width:
            min_width=point[1]
        if point[2]>max_height:
            max_height=point[2]
        if point[2]<min_height:
            min_height=point[2]
    min_width = min_width-50
    max_width = max_width+50
    min_height = min_height-50
    max_height = max_height+50
    cropped_image = image[min_height:max_height, min_width:max_width]
    cv2.imwrite('/home/lvizcaino/code/ManuelLavinSLB/only-hands/data/cropped_hand_with_keypoints.png', cropped_image)
    raw_image = cv2.imread('/home/lvizcaino/code/ManuelLavinSLB/only-hands/data/input_picture.png', cv2.IMREAD_COLOR)
    raw_image = tracker.handsFinder(raw_image, draw=False)
    raw_cropped_image = raw_image[min_height:max_height, min_width:max_width]
    cv2.imwrite('/home/lvizcaino/code/ManuelLavinSLB/only-hands/data/cropped_hand_without_keypoints.png', raw_cropped_image)
