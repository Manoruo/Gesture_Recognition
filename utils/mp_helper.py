
from utils.config import *
import cv2 
import numpy as np


def mediapipe_detection(image, model):
    
    """
    Perform facial, pose, and hand landmark detection on the provided image using the given Mediapipe model.

    Parameters:
    - image (numpy.ndarray): The input image for detection. (should be BGR)
    - model (Mediapipe Holistic model): The Mediapipe Holistic model for landmark detection.

    Returns:
    - image (numpy.ndarray): Annotated image with landmarks drawn.
    - results (Mediapipe Holistic results): Detection results containing facial, pose, and hand landmarks.
    """
    
    # update image properties so we can process it 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # To improve performance, optionally mark the image as not writeable to pass by reference.
    
    # process image 
    results = model.process(image)
    
    # set varaibles back 
    image.flags.writeable = True 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results 


def draw_landmarks(image, results):
    """
    Draw landmarks and connections on the given image for face, pose, and both left and right hands.

    Parameters:
    - image (numpy.ndarray): The input image we will draw the landmarks on 
    - results (Mediapipe Holistic results): Detection results containing facial, pose, and hand landmarks.

    Returns:
    - None
    """
    
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), # landmark drawing color
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)) # connection color 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4), 
                               mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), 
                               mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                               mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def get_keypoints(result):
    
    """
    Extract the 3D coordinates of left and right hand landmarks, pose landmarks, and face landmarks from a single image
    and return this in a flattened array to feed into our model
    Parameters:
    - result (Mediapipe Holistic results): Detection results containing facial, pose, and hand landmarks.

    Returns:
    - keypoints (numpy.ndarray): A flattened array containing the 3D coordinates of all detected landmarks.
    """
    
    
    # it is important to flatten these as it will be passed into our model
    lh_kp = np.array([[res.x, res.y, res.z]  for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(HAND_LANDMARKS)
    rh_kp = np.array([[res.x, res.y, res.z]  for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(HAND_LANDMARKS)
    pos_kp = np.array([[res.x, res.y, res.z, res.visibility]  for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(POS_LANDMARKS)
    face_kp = np.array([[res.x, res.y, res.z]  for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(FACE_LANDMARKS)

    # concat all our data into 1 flat array
    return np.concatenate([lh_kp, rh_kp, pos_kp, face_kp])
