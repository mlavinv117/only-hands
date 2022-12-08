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

    st.warning("**FACT**: Around 70 million people are deaf globally.")

    st.header('The challange')

    st.caption("""
               Creating a Machine Learning model that detects the signs for the 26 different letters of the alphabet is
               such a challenging task, that the Sign Language MNIST dataset is today a popular benchmark for image-based
               machine learning methods.

                The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes
                of letters (excluding J and Z which require motion). Each training and test case represents a label (0-25) as
                a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions).
                The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST
                but otherwise similar with a header row of label, pixel1,pixel2….pixel784 which represent a single 28x28 pixel image
                with grayscale values between 0-255.

                Can we build a Machine Learning model that uses the MNIST dataset to train to detect the 24 letters of the sign language
                that don't require movement? Can it generalize well to real life images of hands? This was our first step in the long
                road of creating a useful app to learn!
               """)

    st.header('Some concepts')

    tab3, tab4, tab5, tab6, tab7 = st.tabs(['Machine Learning (ML)', 'Supervised Learning', 'Deep Learning (DL)', 'Neural Network (NN)', 'Convolutional Neural Network (CNN)'])

    with tab3:

        st.info("""
            The use and development of computer systems that are able to learn and adapt without following explicit instructions,
                by using algorithms and statistical models to analyse and draw inferences from patterns in data.
                """)

    with tab4:

        st.info("""
            A type of ML algoriths that learn from examples of data that are already classified or of which output is known. For example: \n
            -Phots of cat and dogs, that in the name of the picture we specify the class (dog01.png, dog02.png, etc.)
            -The different features of a house that has an impact in its price (#bedrooms, #bathrooms, zone, etc.) and its actual price.
                """)

    with tab5:

        st.info("""
            Refers to a specific type of ML models, which are more robust, require more computational power, and can perform even more difficult tasks
            than conventional ML models. They use advanced algorithms that consist of a concatenation of several linear regressions, activation functions,
            and several other elements that assist on adding complexity. The full stack of these components is known as Neural Network (NN). When a model uses
            one or several NN to perform its task, it is said that this ML model is a Deep Learning model.
                """)
