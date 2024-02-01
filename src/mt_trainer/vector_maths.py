'''
  Utility functions for performing 3d vector maths operations
'''

import math

def landmark_to_vector(landmark):
  return (landmark.x, landmark.y, landmark.z)

def vector_between(vector1, vector2):
  '''
    returns x,y,z co-ordinates of the vector from
    vector1 to vector2
  '''
  return (vector2[0] - vector1[0], vector2[1] - vector2[1], vector2[2] - vector2[1])

def vector_mod(vector):
  '''
    return the modulus of the given vector
  '''
  return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

def dot(vector1, vector2):
  return vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2]

def angle_between(vector1, vector2, vector3):
    '''
    This function calculates angle between three different vectors at the 
    point vector2.
    
    Args:
        vector1: x,y and z coordinates of vector 1
        vector2: x,y and z coordinates of vector 2.
        vector3: x,y and z coordinates of vector 3.
    Returns:
        angle: The calculated angle between the three vectors.

    '''
    # Calculate the angle between the three points
    # as acos( (3-2).(1-2) / (mod(3-2) * mod(1-2)) )
    v23 = vector_between(vector2, vector3)
    v21 = vector_between(vector2, vector1)
    
    radians = math.acos( dot(v23, v21) / (vector_mod(v23) * vector_mod(v21)) )
    return math.degrees(radians)
    