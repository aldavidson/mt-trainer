import pytest

from mt_trainer.camera import Camera

def test_project_3d_points_returns_an_array_of_2d_points_when_given_an_array_of_3d_points():
    camera = Camera(image_width=100, image_height=200)
    assert camera.project_3d_points([(0, 0, 0)]) == [(50, 100)]
    
def test_project_3d_point_returns_a_2d_point_when_given_a_3d_point():
    camera = Camera(image_width=100, image_height=200)
    assert camera.project_3d_point((0, 0, 0)) == (50, 100)
