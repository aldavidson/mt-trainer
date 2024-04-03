import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

import mediapipe as mp
import numpy as np
import time

from PIL import Image, ImageDraw, ImageFont
from mediapipe.framework.formats.landmark_pb2 import LandmarkList, NormalizedLandmarkList
from typing import List, Mapping, Optional, Tuple, Union

class GraphPlotter:
  
    def __init__(self, figure=None, axes=None) -> None:
        self.figure = figure
        self.axes = axes
        
        mplstyle.use('fast')
        pass
  
    def create_figure(self,
                      width: int = 512,
                      height: int = 256):
        self.figure = plt.figure(figsize=(width/100, height/100),
                                 constrained_layout=True,
                                 frameon=False)
        return self.figure
    
    def create_axes(self,
                    elevation: int = 10,
                    azimuth: int = 30,
                    ):
        self.axes = plt.axes(projection='3d')
        self.axes.view_init(elev=elevation, azim=azimuth)
        return self.axes
    
    def cleanup(self):
      plt.close(self.figure)
      
    def grab_image(self):
        '''
          Grab the 3d plot figure and convert it to an image
        '''
        # fig, _ax = plt.subplots()
        # import pdb
        # pdb.set_trace()
        
        start = time.time()
        self.figure.canvas.draw()
        print('drawn canvas in ', time.time() - start, 's')
        
        start = time.time()
        buffer = self.figure.canvas.tostring_rgb()
        print('converted tostring_rgb in ', time.time() - start, 's')
        
        width, height = self.figure.canvas.get_width_height()
        
        start = time.time()
        pil_image = Image.frombytes("RGB", (width, height), buffer)
        print('Image.frombytes in ', time.time() - start, 's')
        
        # start = time.time()
        # plt.close(self.figure)
        # print('plt.close() in ', time.time() - start, 'ms')
        
        start = time.time()
        array = np.array(pil_image)
        print('converted to np.array in ', time.time() - start, 's')
        
        return array
      
    def plot_3d_landmarks_on_image(self,
                                   image: np.ndarray,
                                   landmark_list: NormalizedLandmarkList,
                                   connections: Optional[List[Tuple[int, int]]] = mp.solutions.pose.POSE_CONNECTIONS,
                                  ):
        
        plotted_landmarks = {}
        # skip landmarks that either aren't present, or aren't visible
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and
                landmark.visibility < mp.solutions.drawing_utils._VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < mp.solutions.drawing_utils._PRESENCE_THRESHOLD)):
                continue
            # project from normalised world co-ords (-1:1, -1:1, -1:1)
            # to image-space 2d co-ords
          
        return image
    
    def plot_3d_landmarks(self, 
                          landmark_list: NormalizedLandmarkList,
                          connections: Optional[List[Tuple[int, int]]] = mp.solutions.pose.POSE_CONNECTIONS,
                          landmark_drawing_spec: mp.solutions.drawing_utils.DrawingSpec = mp.solutions.drawing_utils.DrawingSpec(
                              color=mp.solutions.drawing_utils.RED_COLOR, thickness=5),
                          connection_drawing_spec: mp.solutions.drawing_utils.DrawingSpec = mp.solutions.drawing_utils.DrawingSpec(
                              color=mp.solutions.drawing_utils.BLACK_COLOR, thickness=5),
                          ):
        """Plot the landmarks and the connections in matplotlib 3d.

        Args:
            landmark_list: A normalized landmark list proto message to be plotted.
            connections: A list of landmark index tuples that specifies how landmarks to
            be connected.
            landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
            drawing settings such as color and line thickness.
            connection_drawing_spec: A DrawingSpec object that specifies the
            connections' drawing settings such as color and line thickness.
            elevation: The elevation from which to view the plot.
            azimuth: the azimuth angle to rotate the plot.

        Raises:
            ValueError: If any connection contains an invalid landmark index.
        """
        if not landmark_list:
            return
        
        
        plotted_landmarks = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and
                landmark.visibility < mp.solutions.drawing_utils._VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < mp.solutions.drawing_utils._PRESENCE_THRESHOLD)):
                continue
            
            self.axes.scatter3D(
                xs=[-landmark.z],
                ys=[landmark.x],
                zs=[-landmark.y],
                color=mp.solutions.drawing_utils._normalize_color(landmark_drawing_spec.color[::-1]),
                linewidth=landmark_drawing_spec.thickness)
            plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
            
        if connections:
            num_landmarks = len(landmark_list.landmark)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                    f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                    landmark_pair = [
                        plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                    ]
                    self.axes.plot3D(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                        color=mp.solutions.drawing_utils._normalize_color(connection_drawing_spec.color[::-1]),
                        linewidth=connection_drawing_spec.thickness)
        