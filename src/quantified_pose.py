import mediapipe as mp
from .vector_maths import *

class QuantifiedPose:
  mp_pose = mp.solutions.pose
  
  ANGLE_LANDMARKS = {
    "left_ankle_extension": (mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
                  mp_pose.PoseLandmark.LEFT_ANKLE.value,
                  mp_pose.PoseLandmark.LEFT_KNEE.value),
    
    "left_knee_extension": (mp_pose.PoseLandmark.LEFT_ANKLE.value,
                  mp_pose.PoseLandmark.LEFT_KNEE.value,
                  mp_pose.PoseLandmark.LEFT_HIP.value),
    
    "left_hip_elevation": (mp_pose.PoseLandmark.LEFT_KNEE.value,
                  mp_pose.PoseLandmark.LEFT_HIP.value,
                  mp_pose.PoseLandmark.LEFT_SHOULDER.value),
    
    "left_hip_abduction": (mp_pose.PoseLandmark.LEFT_KNEE.value,
                  mp_pose.PoseLandmark.LEFT_HIP.value,
                  mp_pose.PoseLandmark.RIGHT_HIP.value),
    
    "left_shoulder_elevation": (mp_pose.PoseLandmark.LEFT_HIP.value,
                  mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                  mp_pose.PoseLandmark.LEFT_ELBOW.value),
  
    "left_shoulder_abduction": (mp_pose.PoseLandmark.LEFT_ELBOW.value,
                  mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                  mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
    
    "left_elbow_extension": (mp_pose.PoseLandmark.LEFT_WRIST.value,
                  mp_pose.PoseLandmark.LEFT_ELBOW.value,
                  mp_pose.PoseLandmark.LEFT_SHOULDER.value),
    
    "right_ankle_extension": (mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
                  mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                  mp_pose.PoseLandmark.RIGHT_KNEE.value),
    
    "right_knee_extension": (mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                  mp_pose.PoseLandmark.RIGHT_KNEE.value,
                  mp_pose.PoseLandmark.RIGHT_HIP.value),
    
    "right_hip_elevation": (mp_pose.PoseLandmark.RIGHT_KNEE.value,
                  mp_pose.PoseLandmark.RIGHT_HIP.value,
                  mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
    
    "right_hip_abduction": (mp_pose.PoseLandmark.RIGHT_KNEE.value,
                  mp_pose.PoseLandmark.RIGHT_HIP.value,
                  mp_pose.PoseLandmark.LEFT_HIP.value),
    
    "right_shoulder_elevation": (mp_pose.PoseLandmark.RIGHT_HIP.value,
                  mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                  mp_pose.PoseLandmark.RIGHT_ELBOW.value),
  
    "right_shoulder_abduction": (mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                  mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                  mp_pose.PoseLandmark.LEFT_SHOULDER.value),
    
    "right_elbow_extension": (mp_pose.PoseLandmark.RIGHT_WRIST.value,
                  mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                  mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
  }


  def __init__(self, world_landmarks, image_landmarks, angles={}):
    self.world_landmarks = world_landmarks
    self.image_landmarks = image_landmarks
    if bool(angles):
      self.angles = angles
    else:
      self.angles = self.calculate_angles()
  
  def calculate_angles(self):
    angles = {}
    for angle in self.ANGLE_LANDMARKS:
      landmarks = map( lambda l: landmark_to_vector(self.world_landmarks.landmark[l]), self.ANGLE_LANDMARKS[angle])
      angles[angle] = angle_between( *landmarks )
    return angles
