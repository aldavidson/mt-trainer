import pytest

from src.mt_trainer.vector_maths import *

class MockLandmark:
  ''' We don't need to import the actual MediaPipe Landmark class,
      it comes with too many other dependencies. We just need a class
      that has x y and z methods
  '''
  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z


def test_landmark_to_vector_returns_a_3d_vector():
  landmark = MockLandmark(0.1, 0.2, 0.3)
  
  assert landmark_to_vector(landmark) == (0.1, 0.2, 0.3)

def test_vector_between_returns_the_vector_from_a_to_b():
  vector_a = (1, 2, 3)
  vector_b = (4, 5, -6)
  
  assert vector_between(vector_a, vector_b) == (3, 3, -9)
  
def test_vector_mod_returns_the_modulus():
  vector = (3,4)
  assert vector_mod(vector) == 5
  
  vector = (5, -4, 2)
  assert vector_mod(vector) == math.sqrt(45)

def test_dot_returns_dot_product():
  vector_a = (1, 2, 3)
  vector_b = (4, 5, -6)
  assert dot(vector_a,  vector_b) == (1*4 + 2*5 + 3*-6)

def test_angle_between_returns_degrees_between_given_vectors():
  vector_a = (0,1)
  vector_b = (1,0)
  assert angle_between(vector_a, vector_b) == 90
  
  vector_a = (1,1)
  vector_b = (1,0)
  assert round(angle_between(vector_a, vector_b),1) == 45.0
  
  vector_a = (0,1,0)
  vector_b = (1,0,0)
  assert angle_between(vector_a, vector_b) == 90


def test_variance_returns_sqrt_of_sum_of_squared_differences():
  vector_a = (0,1)
  vector_b = (1,0)
  assert variance(vector_a, vector_b) == math.sqrt(2)
  
  vector_a = (3,4,5)
  vector_b = (1,-1, -2)
  
  assert variance(vector_a, vector_b) == math.sqrt(-2*-2 + -5*-5 + -7*-7)
  
  
  
   
  
  
  
