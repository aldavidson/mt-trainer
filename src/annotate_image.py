#!/usr/bin/python
'''
    Estimates the pose of a single person
    in the given image, outputs an annotated
    image to the given file path, and optionally
    plots the landmarks in matplot3d
'''

# cv2 - computer vision lib
# mediapip - google's toolkit for applying AI to media
import argparse
import cv2
import mediapipe as mp
import numpy as np
import os

from mt_trainer.frame_processor import FrameProcessor


def insert_suffix_before_extension(path, suffix):
    ''' Insert the given suffix before the file extension og the given path '''
    root, ext = os.path.splitext(path)
    out_path = root + suffix
    if ext is not None:
        out_path += ext
    return out_path


def default_output_file_path(input_file_path):
    ''' Construct a default output file path from the given input file path '''
    return insert_suffix_before_extension(input_file_path, '-output')


parser = argparse.ArgumentParser(
                    prog='MT-Trainer static image demo',
                    description='Estimates the pose of a single person'
                                'in the given image, outputs an annotated'
                                'image to the given file path, and optionally'
                                'plots the landmarks in matplot3d'
)
parser.add_argument('input_file')
parser.add_argument('--output-file', '-o',
                    dest='output_file')
parser.add_argument('--plot-3d', '-3d',
                    dest='plot_3d', default='false', choices=['false', 'true'])
parser.add_argument('-dc', '--min-detection-confidence',
                    dest='min_detection_confidence',
                    type=float, default=0.5)
parser.add_argument('-tc', '--min-tracking-confidence',
                    dest='min_tracking_confidence',
                    type=float, default=0.5)
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file or default_output_file_path(input_file)


# read the image
mp_image = mp.Image.create_from_file(input_file)

# convert the image to the right format for pose recognition (RGB)
rgb_image = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_BGR2RGB)

processor = FrameProcessor(
    min_detection_confidence=args.min_detection_confidence,
    min_tracking_confidence=args.min_tracking_confidence)


# detect & quantify the pose
pose = processor.quantify_pose(rgb_image)

# Draw landmarks on the image itself
# Returns a copy - the mp_image.numpy_view() is immutable
# converted_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
rgb_image_with_landmarks = processor.draw_landmarks(pose.image_landmarks,
                                                    rgb_image)

# render the body angles to a separate panel
rgb_panel = processor.render_angles(pose)

# Combine the two images into one
combined_image = processor.append_image(rgb_image_with_landmarks, rgb_panel)

print('writing annotated image to ', args.output_file)
cv2.imwrite(
    args.output_file,
    combined_image
)

# plot the pose as a connected skeleton in matlib3d
if args.plot_3d == 'true':
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing.plot_landmarks(pose.world_landmarks, mp_pose.POSE_CONNECTIONS)
