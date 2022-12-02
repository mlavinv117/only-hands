import cv2
import mediapipe as mp
import pandas as pd
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from tensorflow.keras import models
import numpy as np
import streamlit as st

@st.cache
def load_model():
    model = models.load_model('models/NN_from_keypoints')
    return model

def keypoints_preprocessor(keypoints):
    data = {}
    hs = []
    ws = []
    i=0
    for keypoint in keypoints:
        data[str(i) + '_h'] = [keypoint[1]]
        data[str(i) + '_w'] = [keypoint[2]]
        ws.append(keypoint[1])
        hs.append(keypoint[2])
        i+=1
    min_h = min(hs)
    avg_w = int(round(sum(ws)/len(ws),0))
    data_df = pd.DataFrame.from_dict(data)
    return data_df, avg_w, min_h

def prediction_postprocessor(prediction):
    nums_to_letters = {
        0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',
        7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',
        13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',
        19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',
    }
    max_pred = np.argmax(prediction, axis=1)[0]
    max_prob = np.max(prediction)
    print(max_prob)
    y_pred = nums_to_letters[max_pred]
    if max_prob>=0.70:
        return y_pred
    if max_prob<0.70:
        return None

class handTracker(VideoTransformerBase):
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

    def recv(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        frame = self.handsFinder(frame)
        keypoints = self.positionFinder(frame)
        if len(keypoints)==21:
            keypoints, avg_w, min_h = keypoints_preprocessor(keypoints)
            model = load_model()
            prediction = model.predict(keypoints)
            y_pred = prediction_postprocessor(prediction)
            if y_pred:
                frame = cv2.putText(frame,
                                    y_pred,
                                    org = (avg_w, min_h-25),
                                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale = 1,
                                    color = (255, 0, 0),
                                    thickness = 2,)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")
