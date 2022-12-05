import cv2
import mediapipe as mp
import os
import pandas as pd

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
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

mode = 'test'
path = '/Users/manuel/Pictures/raw/' + mode + '/'

keypoints_list = []
letters_list = []
file_paths = []
for f in os.listdir(path):
    if os.path.isdir(path+f):
        for pic in os.listdir(path+f):
            pic_path = path+f+'/'+pic
            picture = cv2.imread(pic_path)
            keypoints = tracker.positionFinder(tracker.handsFinder(picture))
            if len(keypoints) == 21:
                keypoints_list.append(keypoints)
                letters_list.append(f)
                file_paths.append('/content/drive/MyDrive/Louder_Hands/data/own_data/' + mode + '/' + f + '/' + pic)

keypoints_df = pd.DataFrame.from_dict({'path':file_paths, 'keypoints':keypoints_list, 'letters':letters_list})
keypoints_df.to_csv('keypoints_' + mode + '_raw.csv', index=False)
