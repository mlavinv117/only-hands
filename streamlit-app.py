import streamlit as st
import cv2
import av
import mediapipe as mp
from only_hands import handTracker
from only_hands import handTracker_nodraw
from only_hands import keypoints_preprocessor
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from tensorflow.keras import models

st.set_page_config(page_title="Only Hands",
                   page_icon="👌",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)

with st.sidebar:
    keypoints_checkbox = st.checkbox('Show keypoints')
    model_options = st.selectbox(
    'Select a model',
    ('NN from keypoints'))

#st.set_page_config(layout='wide')
col1, col2, col3 = st.columns(3)

#Header with 3 columns to center image
with col1:
    st.write(' ')

with col2:
    st.image('data/only_hands_logo.png')

with col3:
    st.write(' ')


tab1, tab2 = st.tabs(['App', 'Reference'])

with tab1:

    col4, col5 = st.columns(2)

    with col4:

        RTC_CONFIGURATION = RTCConfiguration(
            {
                "iceServers": [{
                    "urls": ["turn:openrelay.metered.ca:80"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                    }]
                }
            )
        if keypoints_checkbox:
            webrtc_ctx = webrtc_streamer(
                key="WYH",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=handTracker,
                async_processing=True,)
        else:
            webrtc_ctx = webrtc_streamer(
                key="WYH",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=handTracker_nodraw,
                async_processing=True,)
    with col5:
        st.image('data/amer_sign2.png')
with tab2:
     st.image('data/amer_sign2.png')
