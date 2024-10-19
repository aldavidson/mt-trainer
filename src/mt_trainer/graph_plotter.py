import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

import cv2
import mediapipe as mp
import numpy as np
import time
import math

from PIL import Image, ImageDraw, ImageFont
from mediapipe.framework.formats.landmark_pb2 import LandmarkList, NormalizedLandmarkList
from typing import List, Mapping, Optional, Tuple, Union

from mt_trainer.camera import Camera
from . import vector_maths

class GraphPlotter:
  
    def __init__(self, figure=None, axes=None) -> None:
        self.figure = figure
        self.axes = axes
        
        mplstyle.use('fast')
        pass
    
    def cleanup(self):
      pass
      
    def plot_3d_landmarks_on_image(self,
                                   image: np.ndarray,
                                   landmark_list: NormalizedLandmarkList,
                                   connections: Optional[List[Tuple[int, int]]] = mp.solutions.pose.POSE_CONNECTIONS,
                                   camera: Camera = None,
                                  ):

        camera = camera or Camera(image_width=image.shape[1],
                                  image_height=image.shape[0])
        points_2d = {}
        # skip landmarks that either aren't present, or aren't visible
        for idx, landmark in enumerate(landmark_list.landmark):
          
            if ((landmark.HasField('visibility') and
                landmark.visibility < mp.solutions.drawing_utils._VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < mp.solutions.drawing_utils._PRESENCE_THRESHOLD)):
                continue

            # project from normalised world co-ords (-1:1, -1:1, -1:1)
            # to image-space 2d co-ords
            point_2d = camera.project_3d_point(vector_maths.landmark_to_vector(landmark))
            points_2d[idx] = point_2d

            image = cv2.circle(image, point_2d, 4, (0, 0, 0), -1)
        
        if connections:
            num_landmarks = len(landmark_list.landmark)
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                    f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in points_2d and end_idx in points_2d:
                    landmark_pair = [
                        points_2d[start_idx], points_2d[end_idx]
                    ]
                    
                    cv2.line(image, landmark_pair[0], landmark_pair[1], (64, 64, 64), 1)
            
        return image
