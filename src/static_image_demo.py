#!/usr/bin/python

# cv2 - computer vision lib
# mediapip - google's toolkit for applying AI to media
import argparse
import cv2
import mediapipe as mp
import numpy as np
import os

from frame_processor import FrameProcessor

# shorthands for lib classes
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def default_output_file_path(input_file):
  root, ext = os.path.splitext(input_file)
  out_path = root + '-output'
  if ext != None:
    out_path += ext
  return out_path

parser = argparse.ArgumentParser(
                    prog='MT-Trainer static image demo',
                    description='Estimates the pose of a single person in the given image, outputs an annotated image to the given file path, and optionally plots the landmarks in matplot3d')
parser.add_argument('input_file')
parser.add_argument('--output-file', '-o', dest='output_file') 
parser.add_argument('--plot-3d', '-3d', dest='plot_3d', default='false', choices=['false','true']) 

args = parser.parse_args()
input_file=args.input_file
output_file=args.output_file or default_output_file_path(input_file)


# read the image
mp_image = mp.Image.create_from_file(input_file)

# convert the image to the right format for pose recognition (RGB)
rgb_image = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_BGR2RGB)

processor = FrameProcessor()

# detect the pose
results = processor.pose_landmarks(rgb_image)

# Draw landmarks on the image itself
# Returns a copy - the mp_image.numpy_view() is immutable
# converted_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
rgb_image_with_landmarks = processor.draw_landmarks(results.pose_landmarks, rgb_image)

# Save the annotated image
print('writing annotated image to ', output_file)
cv2.imwrite(output_file, rgb_image_with_landmarks)

# plot the pose as a connected skeleton in matlib3d
if args.plot_3d == 'true':
  mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)