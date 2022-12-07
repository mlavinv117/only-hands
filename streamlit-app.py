import streamlit as st
import cv2
import av
import mediapipe as mp
import only_hands
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from tensorflow.keras import models

if 'word' not in st.session_state:
    st.session_state['word'] = 'a'

st.set_page_config(page_title="Only Hands",
                   page_icon="👌",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)

with st.sidebar:
    keypoints_checkbox = st.checkbox('Show keypoints', value=True)
    model_options = st.selectbox(
    'Select a model',
    ('NN from keypoints','Hands Only Resnet 256'))

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

    col4, col5 = st.columns([3,2])

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
        if keypoints_checkbox and model_options=='NN from keypoints':
            webrtc_ctx = webrtc_streamer(
                key="WYH",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=only_hands.handTracker,
                video_frame_callback=None,
                async_processing=True,)
        elif not keypoints_checkbox and model_options=='NN from keypoints':
            webrtc_ctx = webrtc_streamer(
                key="WYH",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=only_hands.handTracker_nodraw,
                async_processing=True,)

        elif not keypoints_checkbox and model_options=='Hands Only Resnet 256':
            webrtc_ctx = webrtc_streamer(
                key="WYH",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=only_hands.handTracker_nodraw_CNN,
                async_processing=True,)

    with col5:
        while webrtc_ctx.state.playing:
            with only_hands.lock:
                word = only_hands.word_container["word"]
            if word is None:
                continue
            st.write(word)

with tab2:
     st.image('data/amer_sign2.png')
