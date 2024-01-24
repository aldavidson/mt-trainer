import mediapipe as mp
import vector_maths

class QuantifiedPose:
    mp_pose = mp.solutions.pose
  
    ANGLE_LANDMARKS = {
        "left_ankle_extension": (mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
                                 mp_pose.PoseLandmark.LEFT_ANKLE.value,
                                 mp_pose.PoseLandmark.LEFT_KNEE.value),

        "left_knee_extension": (mp_pose.PoseLandmark.LEFT_ANKLE.value,
                                mp_pose.PoseLandmark.LEFT_KNEE.value,
                                mp_pose.PoseLandmark.LEFT_HIP.value),

        "left_hip_extension": ( mp_pose.PoseLandmark.LEFT_KNEE.value,
                                mp_pose.PoseLandmark.LEFT_HIP.value,
                                mp_pose.PoseLandmark.LEFT_SHOULDER.value),

        "left_hip_abduction": ( mp_pose.PoseLandmark.LEFT_KNEE.value,
                                mp_pose.PoseLandmark.LEFT_HIP.value,
                                mp_pose.PoseLandmark.RIGHT_HIP.value),

        "left_shoulder_elevation": (mp_pose.PoseLandmark.LEFT_HIP.value,
                                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                    mp_pose.PoseLandmark.LEFT_ELBOW.value),

        "left_shoulder_abduction": (mp_pose.PoseLandmark.LEFT_ELBOW.value,
                                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                    mp_pose.PoseLandmark.RIGHT_SHOULDER.value),

        "left_elbow_extension": ( mp_pose.PoseLandmark.LEFT_WRIST.value,
                                  mp_pose.PoseLandmark.LEFT_ELBOW.value,
                                  mp_pose.PoseLandmark.LEFT_SHOULDER.value),

        "right_ankle_extension": (mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
                                  mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                                  mp_pose.PoseLandmark.RIGHT_KNEE.value),

        "right_knee_extension":  (mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                                  mp_pose.PoseLandmark.RIGHT_KNEE.value,
                                  mp_pose.PoseLandmark.RIGHT_HIP.value),

        "right_hip_extension": (mp_pose.PoseLandmark.RIGHT_KNEE.value,
                                mp_pose.PoseLandmark.RIGHT_HIP.value,
                                mp_pose.PoseLandmark.RIGHT_SHOULDER.value),

        "right_hip_abduction": (mp_pose.PoseLandmark.RIGHT_KNEE.value,
                                mp_pose.PoseLandmark.RIGHT_HIP.value,
                                mp_pose.PoseLandmark.LEFT_HIP.value),

        "right_shoulder_elevation":  (mp_pose.PoseLandmark.RIGHT_HIP.value,
                                      mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                      mp_pose.PoseLandmark.RIGHT_ELBOW.value),

        "right_shoulder_abduction":  (mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                                      mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                      mp_pose.PoseLandmark.LEFT_SHOULDER.value),

        "right_elbow_extension": (mp_pose.PoseLandmark.RIGHT_WRIST.value,
                                  mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                                  mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
    }

    def __init__(self, world_landmarks, image_landmarks, angles=None):
        self.world_landmarks = world_landmarks
        self.image_landmarks = image_landmarks
        if angles is not None:
            self.angles = angles
        else:
            self.angles = self.calculate_angles()
   
    def calculate_angles(self):
        angles = {}
        for angle, landmark_names in self.ANGLE_LANDMARKS.items():
            landmarks = map( 
                lambda l: vector_maths.landmark_to_vector(
                            self.world_landmarks.landmark[l]
                          ),
                landmark_names
            )
            angles[angle] = vector_maths.angle_between(*landmarks)
        return angles

    def rounded_angles(self):
        return dict((k, round(v, 0)) for k, v in self.angles.items())

    def length_of_longest_label(self):
        return max(len(key) for key, _value in self.ANGLE_LANDMARKS.items())
