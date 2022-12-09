import streamlit as st
import cv2
import av
import mediapipe as mp
import only_hands
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from tensorflow.keras import models
import platform
import pandas as pd

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

    st.header('The challenge')

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

    with st.expander('Check mediapipe in action!'):
        tab8, tab9, tab10, tab11 = st.tabs(['The input image', 'The output of a cropped image of a hand', 'The output of the keypoints', 'Keypoints reference'])

        with tab8:

            st.image('data/input_picture.png')

        with tab9:

            st.image('data/cropped_hand_without_keypoints.png')

        with tab10:

            col6, col7, col8, col9 = st.columns([1,2,2,1])

            with col7:

                st.write('Keypoints over the picture:')
                st.image('data/cropped_hand_with_keypoints.png')

            with col8:

                st.write('Their numerical representation:')
                st.write(
                         """
                         [[0, 196, 316], [1, 242, 308], [2, 282, 282], [3, 309, 255], [4, 332, 234], [5, 257, 210], [6, 271, 168], [7, 278, 141], [8, 283, 117],
                         [9, 235, 199], [10, 251, 149], [11, 262, 119], [12, 269, 92], [13, 210, 198], [14, 212, 147], [15, 214, 116], [16, 216, 89], [17, 185, 206],
                         [18, 187, 164], [19, 188, 139], [20, 191, 114]]
                         """
                         )



        with tab11:

            st.image('data/keypoints_reference.png')

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
        keypoints = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                keypoints.append([id,cx,cy])
            if draw:
                cv2.circle(image,(cx,cy), 5 , (255,0,255), cv2.FILLED)

        return keypoints
    """
    st.code(mediapipe_code, language='python')

    st.caption("""
               Above is the main Python class through which we implemented mediapipe package to extract both a picture of a hand and its corresponding keypoints.
               It is composed by a constructor (__init__) and two functions: handsFinder and positionFinder. Both functions take a picture as an input.
               The first function, handsFinder, will return the input image with the keypoints on top of the found image. The second function, positionFinder, will
               return a list of the keypoint found and the height, width coordinates for each of them.

               """)

    st.subheader('The benchmark to beat')

    st.caption("""
               Whenever we deal with a ML model, we have to stablish a benchmark to which we will compare if our model is performing better or worse. This will give us
               the feedback if all the effort we took into building a model was actually worth, i.e., if we are performing better than the benchmark model, and also
               measure how much better. There are 2 typical benchmark dummy models, depending on the task to perform:\n
               1. The mean of the population if we have a regression task (the average price of a house within our population if we want to predict a price).\n
               2. The most frequent class if we have a classification task (the most frequently used letter in the alphabet).\n
               The most frequent letter in the English language is "E", according to this article (http://norvig.com/mayzner.html).

               """)

    st.header('Taking ownership of the data')

    st.caption("""
               The Sign Language MNIST dataset, as mention before, is a collection of more than 32,000 pictures of all the letters in the sign language, but
               unfortunatelly, each of the pictures has a 28x28 pixels size and a single color channel (gray scale). While it is possible to build a powerful
               model that performs excellent in the test data from this set, it is difficult to believe it can generalize well with real life color pictures of hands,
               taken with modern webcams or cellphones.\n
               That is why we decided to take the training data of our own. We modified the class above and implemented a code that allow us to take n number of pictures
               for a specified letter, save each of them in a specific folder (so each class is properly separated and labelled), and extract the two main features:
               cropped pictures of the hand performing the sign for the letter, and the corresponding list of keypoints/coordinates. \n
               We generated a dataset of nearly 10,000 images, having at least 300 pictures of each sign to train and 100 pictures to test our models.

               """)

    with st.expander("Check the example:"):

        st.image('data/own_dataset_example.PNG')

    st.header('The NN model (only the keypoints!)')

    st.subheader('Data preparation')

    st.caption("""
               The first approach was to start "simple". Through a modifyed implementation of our main class handTracker, wrapped in a function, we first created a Pandas DataFrame
               from the keypoints. Each row of this DataFrame represents the extracted keypoints of each of the 10,000 pictures. The DataFrame has 43 columns: 42 columns that represent the
               X and Y coordinate of the 21 keypoints, and the label of the corresponding letter of the alphabet that is being represented in the hand picture.

               """)

    df_example = pd.read_csv('data/keypoints_dataframe_example.csv')

    st.dataframe(data=df_example)

    st.subheader('Model design')

    nn_code = """
import pandas as pd
from google.colab import drive
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

drive.mount('/content/drive')
path = '/content/drive/MyDrive/unsounded/data/own_data/'

keypoints_df = pd.read_csv('keypoints.csv')

X = keypoints_df.drop(columns=['letter'])
y = keypoints_df['letter']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                y,
                                                test_size=0.3,
                                                random_state=0)

ohe = OneHotEncoder(sparse=False)
y_train_cat = ohe.fit_transform(pd.DataFrame(y_train))
y_test_cat = ohe.transform(pd.DataFrame(y_test))

def initialize_model():

    model = models.Sequential()

    model.add(layers.Normalization(input_dim=42))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(75, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(24, activation='softmax'))

    return model

def compile_model(model):

    model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

    return model

es = EarlyStopping(patience=20, restore_best_weights=True)

model = initialize_model()
model = compile_model(model)

history = model.fit(X_train,
                    y_train_cat,
                    validation_split = 0.2,
                    epochs = 500,
                    callbacks = [es],
                    batch_size= 32,
                    verbose = 1)

results = model.evaluate(X_test, y_test_cat)

print(results)
    """
    st.code(nn_code, language='python')

    #st.warning("Notice the quotes on the word 'simple' on the paragrah above. This was actually not the simplest posible approach. We already had a feeling that")
