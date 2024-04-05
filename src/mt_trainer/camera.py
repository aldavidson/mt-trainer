import numpy as np
import cv2

class Camera:
  
    def __init__(self,
                 focal_length: int = 400,
                 image_width: int = 640,
                 image_height: int = 480,
                 position: np.ndarray = np.array([0,0,2], dtype=np.float32),
                 orientation: np.ndarray = np.array([0,0,0], dtype=np.float32),
                 ):
        self.focal_length = focal_length
        self.image_width = image_width
        self.image_height = image_height
        self.position = position
        self.orientation = orientation

    def intrinsic_matrix(self):
        intrinsic_matrix = np.array([ 
            [self.focal_length, 0, self.image_width/2], 
            [0, self.focal_length, self.image_height/2], 
            [0, 0, 1] 
        ])
        return intrinsic_matrix
    
    def project_3d_point(self, point):
        projected_point = cv2.projectPoints(point, 
                                   self.orientation, self.position.reshape(-1, 1),
                                   self.intrinsic_matrix(), 
                                   None)
      
        # cv2.projectPoints returns a complicated array of arrays
        # we only want the first element (the projected point)
        projected_point = projected_point[0].flatten()

        return ( int(round(projected_point[0], 0)),
          int(round(projected_point[1], 0)) )
    
    def project_3d_points(self, points):
        return [self.project_3d_point(x) for x in points]
