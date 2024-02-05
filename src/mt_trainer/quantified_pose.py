import mediapipe as mp
import mt_trainer.vector_maths as vector_maths
import pdb

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
            if self.world_landmarks is not None:
                self.angles = self.calculate_angles()
   
    def calculate_angles(self):
        angles = {}
        for angle, landmark_names in self.ANGLE_LANDMARKS.items():
            landmarks = list(map( 
                lambda l: vector_maths.landmark_to_vector(
                            self.world_landmarks.landmark[l]
                          ),
                landmark_names
            ))

            angles[angle] = vector_maths.angle_between(
                vector_maths.vector_between(landmarks[1], landmarks[2]),
                vector_maths.vector_between(landmarks[1], landmarks[0]),
            )
        return angles

    def rounded_angles(self):
        return dict((k, round(v, 0)) for k, v in self.angles.items())

    @staticmethod
    def length_of_longest_label():
        return 24
        # return max(
        #     len(key) for key, _ in QuantifiedPose.ANGLE_LANDMARKS.items()
        # )

    def minus(self, other_pose):
        '''
            Subtract the values of angles and both types of landmarks
            of other_pose from this pose, returning a new copy 
        '''
        diff = QuantifiedPose(self.world_landmarks, self.image_landmarks, self.angles)

        for i, landmark in enumerate(diff.world_landmarks.landmark):
            landmark.x -= other_pose.world_landmarks.landmark[i].x
            landmark.y -= other_pose.world_landmarks.landmark[i].z
            landmark.z -= other_pose.world_landmarks.landmark[i].z

        for i, landmark in enumerate(diff.image_landmarks.landmark):
            landmark.x -= other_pose.image_landmarks.landmark[i].x
            landmark.y -= other_pose.image_landmarks.landmark[i].z
            landmark.z -= other_pose.image_landmarks.landmark[i].z

        for key in diff.angles.keys():
            diff.angles[key] -= other_pose.angles[key]

        return diff

    def similarity_to(self, other_pose):
        '''
            Returns the cosine-similarity compared to the other_pose.
            Method:
            Treats the pose's angles as an n-dimensional vector, and
            returns the cosine of the angle between them in that 
            n-dimensional space. 
            Value ranges from -1 (exactly opposing) to +1 (exactly
            the same)
        '''
        vector1 = self.angles.values()
        vector2 = other_pose.angles.values()
        dot12 = vector_maths.dot(vector1, vector2)
        mod1mod2 = (
            vector_maths.vector_mod(vector1) * vector_maths.vector_mod(vector2)
        )
        return dot12 / mod1mod2
    