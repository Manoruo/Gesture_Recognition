import os 
import numpy as np 
import mediapipe as mp 

# Constants 
HAND_LANDMARKS = 21 * 3 # 21 points, 3 dimensions 
POS_LANDMARKS = 33 * 4 # 33 points, 4 dimensions 
FACE_LANDMARKS = 468 * 3 # 468 poitns, 3 dimensions
DATA_LEN = (2 * HAND_LANDMARKS) + POS_LANDMARKS + FACE_LANDMARKS # total length of data when flattened
DATA_PATH = os.path.join('DATA')

# Configurables
TEST_SPLIT = 0.05 # The percent of data dedicated to testing 
NUM_EPOCHS = 250 # the number of epochs to train the model
NUM_EXAMPLES = 30 # the number of examples we collect for each action
SEQUENCE_LENGTH = 30 # the number of frames for each action example 
ACTIONS = np.array(['lift', 'land', 'follow']) # actions that we can detect 
LABEL_MAP = {label:num for num, label in enumerate(ACTIONS)} 

# Mediapipe Models 
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils