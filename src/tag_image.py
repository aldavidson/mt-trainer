#!/usr/bin/python
""" tag_video.py
Given an input file and a technique name, 
saves a JSON representation of the body angles in the
detected pose into the folder named (technique) under the 
given output folder (default: ./poses/training/)
"""
import argparse
import os
import sys

import cv2

from mt_trainer.frame_processor import FrameProcessor
from mt_trainer.quantified_pose import QuantifiedPose


def print_debug_line(*variables):
    ''' Writes the given line to STDOUT if in verbose mode, otherwise no-op '''
    if args.verbose == 'true':
        sys.stdout.write(' '.join([str(var) for var in variables]))


def insert_suffix_before_extension(path, suffix):
    ''' Insert the given suffix before the file extension of the given path '''
    root, ext = os.path.splitext(path)
    out_path = root + suffix
    if ext is not None:
        out_path += ext
    return out_path


def output_file_name(input_path, output_dir):
    return os.path.join(
        output_dir,
        os.path.basename(input_path) + '.json'
    )


parser = argparse.ArgumentParser(
    prog='tag_image.py',
    description=(
        "Estimates the pose of a single person in the given image file, "
        "and saves a JSON representation of the body angles in the "
        "detected pose into the folder named (technique) under the "
        "given output folder (default: ./poses/training/)")
    )

parser.add_argument('input_file')
parser.add_argument('-v', '--verbose',
                    choices=['true', 'false'], default='false', dest='verbose')
parser.add_argument('-t', '--technique',
                    type=str, default='false', dest='technique',
                    choices=QuantifiedPose.TECHNIQUES)
parser.add_argument('-o', '--output-dir',
                    type=str, default='./data/poses/training',
                    dest='output_dir')
parser.add_argument('-dc', '--min-detection-confidence',
                    dest='min_detection_confidence',
                    type=float, default=0.5)
parser.add_argument('-tc', '--min-tracking-confidence',
                    dest='min_tracking_confidence',
                    type=float, default=0.5)

args = parser.parse_args()

processor = FrameProcessor(
    min_detection_confidence=args.min_detection_confidence,
    min_tracking_confidence=args.min_tracking_confidence)

frame = cv2.imread(args.input_file)
if frame is None:
    print("Error opening image file", args.input_file)
    raise TypeError

print_debug_line(' quantifying pose')
pose = processor.quantify_pose(frame)
if pose.angles:
    output_file = output_file_name(
        args.input_file,
        os.path.join(args.output_dir, args.technique)
    )
        
    pose.save_angles(output_file)

    print(' ', output_file, ' - ', os.path.getsize(output_file), ' bytes')
else:
    print('no pose found in image ', args.input_file)

print('All done')
# cleanup
processor.pose_landmarker.close()
