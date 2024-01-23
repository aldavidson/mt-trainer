import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


from quantified_pose import QuantifiedPose
from text_rendering import Cv2TextRenderer, PILTextRenderer

class FrameProcessor:

  def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    self.pose_landmarker = mp.solutions.pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
    
  def pose_landmarks(self, rgb_image):
    return self.pose_landmarker.process(rgb_image)

  def draw_landmarks(self, landmarks, rgb_image):
    '''
      Return a copy of the image with landmarks drawn & connected
      Can only do this on a copy - the mp_image.numpy_view() is immutable
    '''
    rgb_image_copy = np.copy(rgb_image)
    mp.solutions.drawing_utils.draw_landmarks(
      rgb_image_copy,
      landmarks, 
      mp.solutions.pose.POSE_CONNECTIONS,
      landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    
    return rgb_image_copy