#!/usr/bin/python
""" tag_video.py
Given an input file, a technique name, and a list of frames,
saves a JSON representation of the body angles in each of the
given frames into the folder named (technique) under the 
given output folder (default: ./poses/training/)
"""
import argparse
import json
import os
import pdb
import sys

import cv2

from mt_trainer.frame_processor import FrameProcessor
from mt_trainer.quantified_pose import QuantifiedPose


def print_debug_line(*variables):
    ''' Writes the given line to STDOUT if in verbose mode, otherwise no-op '''
    if args.verbose == 'true':
        sys.stdout.write(' '.join([str(var) for var in variables]))


def insert_suffix_before_extension(path, suffix):
    ''' Insert the given suffix before the file extension og the given path '''
    root, ext = os.path.splitext(path)
    out_path = root + suffix
    if ext is not None:
        out_path += ext
    return out_path


def output_file_name(input_path, output_dir, frame_no):
    return os.path.join(
        output_dir,
        os.path.basename(input_path) +
            '-frame-' + str(frame_no) +
            '.json'
    )


parser = argparse.ArgumentParser(
    prog='landmark_video.py',
    description=(
        "Estimates the pose of a single person in the given video file, "
        "and outputs a video annotated with pose landmarks to the given "
        "file path")
    )

parser.add_argument('input_file')
parser.add_argument('-v', '--verbose',
                    choices=['true', 'false'], default='false', dest='verbose')
parser.add_argument('-t', '--technique',
                    type=str, default='false', dest='technique',
                    )
parser.add_argument('-o', '--output-dir',
                    type=str, default='./data/poses/training',
                    dest='output_dir')
parser.add_argument('-f', '--frames',
                    type=str, default='false', dest='frames',
                    required=True,
                    help=("frame(s) to output, separated by commas."
                          "E.g. --frames 27,84,89,212"))
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

cap = cv2.VideoCapture(args.input_file)
if cap.isOpened() is False:
    print("Error opening video stream or file")
    raise TypeError

# sort the frames so that we can step from first to last in a 
# logical iteration
frames = sorted([int(s) for s in args.frames.split(',') if s.strip()])
next_target_frame = frames[0]
last_frame = min(frames[-1], int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

print_debug_line('tagging frames', frames, ' as', args.technique)

for target_frame in frames:
    frame = None

    print_debug_line('target_frame', target_frame)
    
    while (cap.isOpened() and 
           (cap.get(cv2.CAP_PROP_POS_FRAMES) <= target_frame)):

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print_debug_line('skipping frame', frame_number, '\r')
        ret, frame = cap.read()
        if not ret:
            print("Couldn't read frame ", frame_number,
                  "from", args.input_file,
                  "aborting!")
            break

    print_debug_line('Frame', frame_number)

    if frame_number == target_frame:
        print_debug_line(' quantifying frame', frame_number)
        pose = processor.quantify_pose(frame)
        if pose.angles:
            output_file = output_file_name(
                args.input_file,
                os.path.join(args.output_dir, args.technique),
                frame_number)
            
            pose.save_angles(output_file)

            print(' ', output_file, ' - ', os.path.getsize(output_file), ' bytes')
        else:
            print('no pose found in frame ', frame_number, ', skipping')

print('All done')
# cleanup
processor.pose_landmarker.close()
cap.release()
