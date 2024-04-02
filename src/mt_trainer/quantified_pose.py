import mediapipe as mp
from . import vector_maths
import json
import pdb

from google.protobuf.json_format import MessageToDict, ParseDict
from mediapipe.framework.formats.landmark_pb2 import LandmarkList

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

    def __init__(self, world_landmarks=None, image_landmarks=None, angles=None):
        self.world_landmarks = world_landmarks
        self.image_landmarks = image_landmarks
        if angles is not None:
            self.angles = angles
        else:
            if self.world_landmarks is not None:
                self.angles = self.calculate_angles()
   
    def calculate_angles(self):
        '''
            Calculate the body angles from world_landmarks
        '''
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
        ''' The body angles, but rounded to integers '''
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

        if other_pose.world_landmarks and diff.world_landmarks:
            for i, landmark in enumerate(diff.world_landmarks.landmark):
                landmark.x -= other_pose.world_landmarks.landmark[i].x
                landmark.y -= other_pose.world_landmarks.landmark[i].y
                landmark.z -= other_pose.world_landmarks.landmark[i].z
        
        if other_pose.image_landmarks and diff.image_landmarks:
            for i, landmark in enumerate(diff.image_landmarks.landmark):
                landmark.x -= other_pose.image_landmarks.landmark[i].x
                landmark.y -= other_pose.image_landmarks.landmark[i].y
                landmark.z -= other_pose.image_landmarks.landmark[i].z

        for key in diff.angles.keys():
            diff.angles[key] -= other_pose.angles[key]

        return diff

    def plus(self, other_pose):
        '''
            Add the values of angles and both types of landmarks
            of other_pose to this pose, returning a new copy 
        '''
        diff = QuantifiedPose(
            self.world_landmarks,
            self.image_landmarks,
            self.angles
        )

        if not diff.world_landmarks:
            diff.world_landmarks = other_pose.world_landmarks
        else:
            for i, landmark in enumerate(diff.world_landmarks.landmark):
                landmark.x += other_pose.world_landmarks.landmark[i].x
                landmark.y += other_pose.world_landmarks.landmark[i].y
                landmark.z += other_pose.world_landmarks.landmark[i].z
        
        if not diff.image_landmarks:
            diff.image_landmarks = other_pose.image_landmarks
        else:
            for i, landmark in enumerate(diff.image_landmarks.landmark):
                landmark.x += other_pose.image_landmarks.landmark[i].x
                landmark.y += other_pose.image_landmarks.landmark[i].y
                landmark.z += other_pose.image_landmarks.landmark[i].z

        for key in other_pose.angles.keys():
            diff.angles[key] = diff.angles.get(key, 0.0) + other_pose.angles[key]

        return diff
    
    def multiply_by(self, scale_factor=1.0):
        '''
            Multiply the values of angles and both types of landmarks
            of this pose by the given scale_factor.
            Mostly used for averaging a set of poses in training
        '''

        if self.world_landmarks:
            for i, landmark in enumerate(self.world_landmarks.landmark):
                landmark.x *= scale_factor
                landmark.y *= scale_factor
                landmark.z *= scale_factor
        
        if self.image_landmarks:
            for i, landmark in enumerate(self.image_landmarks.landmark):
                landmark.x *= scale_factor
                landmark.y *= scale_factor
                landmark.z *= scale_factor

        for key in self.angles.keys():
            self.angles[key] *= scale_factor

        return self
    
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
        if self.angles and other_pose.angles:
            vector1 = self.angles.values()
            vector2 = other_pose.angles.values()
            dot12 = vector_maths.dot(vector1, vector2)
            mod1mod2 = (
                vector_maths.vector_mod(vector1) * vector_maths.vector_mod(vector2)
            )
            return dot12 / mod1mod2
        else:
            return None

    def load_angles(self, filepath):
        self.angles = json.load(open(filepath, 'r', encoding='utf-8'))
        return self.angles
    
    def save_angles(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.angles, f)

    def save(self, filepath):
        '''
            Saves the angles and landmarks to the given filepath 
            as JSON
        '''
        doc = {
            "angles": self.angles,
            "world_landmarks": MessageToDict(self.world_landmarks),
            "image_landmarks": MessageToDict(self.image_landmarks),
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(doc, f)
            
    @staticmethod
    def load(filepath):
        '''
            Return a new instance initialised with the JSON data
            in the given filepath
        '''
        doc = json.load(open(filepath, 'r', encoding='utf-8'))
        pose = QuantifiedPose(
            ParseDict(doc.get("world_landmarks"), LandmarkList()),
            ParseDict(doc.get("image_landmarks"), LandmarkList()),
            doc.get("angles"),
        )
        return pose
    
    def plot_3d(self):
        '''
            Render world landmarks in 3d using matplotlib
        '''
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.plot_landmarks(self.world_landmarks, mp_pose.POSE_CONNECTIONS)