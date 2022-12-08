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

    with tab6:

        st.info("""
            A NN is a composed by several layers of processing elements. A layer can be composed of one or more 'neuron', which is a
            concatenation of a single linear regression followed by an activation function. This algorithm received the name of 'neuron' since it
            tries to emulate the biological neurons of the brain: they perform a calculation and signal the output to the other neurons of the brain
            (process known as 'synapsis').
                """)

    with tab7:

        st.info("""
            CNNs are a more advanced version of regular NNs, since they can process multi-dimensional arrays, for example images. A CNN model can receive images
            as input and can perform complex tasks like identyfing certain element on it (for example, detect a hand!) and they can also perform
            classification tasks (for example, given a hand sign, classify to which letter of the sign lenguage it corresponds to!).
                """)

    st.header('The approach(es)')

    st.caption("""
               The task we had in hand was, given a picture of a hand sign, determine to which letter of the alphabet it corresponds to. With this in mind,
               and having into consideration the previous concepts, we decided to try 3 approaches:
               1. Build a NN that uses keypoints of a hand to classify the sign into a letter.
               2. Build a CNN that uses a picture of hand to classify the sign into a letter.
               3. Concatenate both the NN from point 1 and the CNN from point 2 into a non-sequential DL model.
               """)
    st.subheader('Wait... keypoints?')

    st.caption("""
               Introducing Mediapipe!\n

               Mediapipe is a powerful library in Python, developed by Google, which is already a DL model that is useful to detect several objects from
               pictures... including HANDS! A mantra in ML is: if somebody has already done it, build on top of it instead of starting from scratch.
               Python, being an Open Source programming language, has a great spirit of sharing and the largest programmers community worlwide!\n

               We decided to use mediapipe library to smash the first part of our problem: given a picture, determine if we have a hand on it and extract it.
               The second part of the problem, given a picture of a hand, determine if it corresponds to a letter of the alphabet in sign language, is what we
               will specifically train our own model to solve. \n

                So, with mediapipe we created a function that gets a picture, and deliver two outputs: the cropped picture of only the hand and
                20 keypoints of the hand, this is, a representation in coordinates (no longer pixels!) of how the palm and fingers are located with a picture from each other.
                The keypoints allow us to transform an image problem (CNN) into a numerical features problem (NN).


               """)
    tab8, tab9, tab10 = st.tabs(['The input image', 'The output of a cropped image of a hand', 'The output of the keypoints'])

    with tab8:

        st.image('data/input_image.png')

    with tab9:

        st.image('data/cropped_hand_without_keypoints.png')

    with tab10:

        st.image('data/cropped_hand_with_keypoints.png')

    st.subheader('Our mediapipe implementation')

    mediapipe_code = """
    class handTracker():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
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
    """
    st.code(mediapipe_code, language='python')

    st.caption("""
               Above is the main Python class through which we implemented mediapipe package to extract both a picture of a hand and its corresponding keypoints.
               It is composed by a constructor (__init__) and two functions: handsFinder and positionFinder. Both functions take a picture as an input.
               The first function, handsFinder, will return the input image with the keypoints on top of the found image. The second function, positionFinder, will
               return a list of the keypoint found and the height, width coordinates for each of them.

               """)
