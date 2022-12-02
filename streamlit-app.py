import streamlit as st
import cv2
import av
import mediapipe as mp
from only_hands import handTracker
from only_hands import keypoints_preprocessor
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from tensorflow.keras import models

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
col4, col5, col6 = st.columns(2)

#Left column to show webcam

with col4:
    pass

with col5:

    RTC_CONFIGURATION = RTCConfiguration(
        {
            "iceServers": [{
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
                }]
            }
        )
    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=handTracker,
        async_processing=True,)

with col6:
    pass
