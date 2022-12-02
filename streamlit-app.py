import streamlit as st
import cv2
import av
import mediapipe as mp
from only_hands import handTracker
from only_hands import keypoints_preprocessor
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from tensorflow.keras import models

model = models.load_model('models/NN_from_keypoints')

st.set_page_config(layout='wide')
col1, col2, col3 = st.columns(3)

#Header with 3 columns to center image
with col1:
    st.write(' ')

with col2:
    st.image('data/only_hands_logo.png')

with col3:
    st.write(' ')

#Main body with 2 columns to split webcam and prediction
col4, col5 = st.columns(2)

#Left column to show webcam
with col4:
    #run = st.checkbox('Run')
    # FRAME_WINDOW = st.image([])
    # camera = cv2.VideoCapture(0)
    # tracker = handTracker()
    # p = st.empty()
    #while run:
    #     _, frame = camera.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frame = tracker.handsFinder(frame)
    #     lmList = tracker.positionFinder(frame)
    #     FRAME_WINDOW.image(frame)
    #     #print(lmList)
    #     if len(lmList)==21:
    #         p.write(keypoints_preprocessor(lmList))
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun1.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    #video_processor_factory=VideoProcessor,
    async_processing=True,
)

#Right column to show prediction
with col5:
    with st.container():
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.markdown("<h1 style='text-align: center; color: grey; vertical-align:middle;'>Translated letter:</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: grey; vertical-align:middle;'>B</h2>", unsafe_allow_html=True)
