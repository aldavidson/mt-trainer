'''
  Utility functions for performing 3d vector maths operations
'''

import math


def landmark_to_vector(landmark):
    ''' convert the given MediaPipe landmark to a vector '''
    return (landmark.x, landmark.y, landmark.z)


def vector_between(vector1, vector2):
    '''
      returns the vector from
      vector1 to vector2
    '''
    return tuple(map(lambda v1, v2: v2-v1, vector1, vector2))


def vector_mod(vector):
    '''
      return the modulus of the given vector
    '''
    return math.sqrt(sum(map(lambda v1: v1*v1, vector)))


def dot(vector1, vector2):
    ''' return the dot product of the two vectors '''
    return sum(map(lambda v1, v2: v1*v2, vector1, vector2))


def angle_between(vector1, vector2):
    '''
      This function calculates angle between two vectors

      Args:
          vector1
          vector2
      Returns:
          angle: The calculated angle between the two vectors.
    '''
    radians = math.acos(
        dot(vector1, vector2) /
        (vector_mod(vector1) * vector_mod(vector2))
    )
    return math.degrees(radians)


def variance(vector1, vector2):
    '''
        Returns the "variance" between the two vectors - e.g.
        sqrt( sum( (x2-x1)**2, (y2-y1)**2, (z2-z1)**2, ... )
    '''
    return math.sqrt(sum(map(lambda i1, i2: (i2-i1)**2, vector1, vector2)))
