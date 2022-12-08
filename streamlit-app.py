import streamlit as st
import cv2
import av
import mediapipe as mp
import only_hands
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from tensorflow.keras import models
import platform

st.set_page_config(page_title="UNS🤏UNDED",
                   page_icon="👂",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)

with st.sidebar:
    keypoints_checkbox = st.checkbox('Show keypoints', value=True)
    model_options = st.selectbox(
    'Select a model',
    ('NN from keypoints','Resnet50 from images','Concatenated model keypoints + images'))

#st.set_page_config(layout='wide')
col1, col2, col3 = st.columns([1,3,1])

#Header with 3 columns to center image
with col1:
    pass

with col2:
    st.image('data/unsounded.png')

with col3:
    pass

tab1, tab2 = st.tabs(['App', 'Reference'])

with tab1:

    col4, col5 = st.columns([3,2])

    with col4:

        if type(platform.processor()) == str:
            RTC_CONFIGURATION = RTCConfiguration(
                {
                    "iceServers": [{
                        "urls": ["stun:stun.l.google.com:19302"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                        }]
                    }
                )
        else:
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
                #video_frame_callback=only_hands.handTracker.callback,
                async_processing=True,)
        elif not keypoints_checkbox and model_options=='NN from keypoints':
            webrtc_ctx = webrtc_streamer(
                key="WYH",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=only_hands.handTracker_nodraw,
                async_processing=True,)

        elif not keypoints_checkbox and model_options=='Resnet50 from images':
            webrtc_ctx = webrtc_streamer(
                key="WYH",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=only_hands.handTracker_image_only,
                async_processing=True,)

        elif not keypoints_checkbox and model_options=='Concatenated model keypoints + images':
            webrtc_ctx = webrtc_streamer(
                key="WYH",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=only_hands.handTracker_concat,
                async_processing=True,)

    with col5:
        st.write('Try to make the signs to create your favorite word!')
        st.image('data/amer_sign2.png')

with tab2:

    st.title('Unsounded by slb')

    st.header('Where it all started...')

    st.caption("""
               Have you ever seen the movie "The sound of metal" by Darius Marder? In one sentence, it is about
               a heavy-metal drummer's life is thrown into freefall when he begins to lose his hearing. The
               struggle of a person losing so important motivated us to develop Unsounded, an app that can teach you
               to learn to spell letters in sign language.
               """)

    st.info("**FACT**: Around 70 million people are deaf globally")
