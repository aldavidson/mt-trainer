#!/usr/bin/python
""" landmark_video.py
Estimates the pose of a single person in the given video file,
and outputs a video annotated with pose landmarks to the given
file path.
Optionally scales the output video and reencodes it with a
given codec
"""

# cv2 - computer vision lib
# mediapipe - google's toolkit for applying AI to media
from __future__ import annotations
import argparse
import os
import sys
from tabnanny import verbose

import cv2
import mediapipe as mp
import numpy as np
from quantified_pose import QuantifiedPose

# shorthands for lib classes
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def default_output_file_path(path):
    """Append -output before the file extension in the given file path"""
    root, ext = os.path.splitext(path)
    out_path = root + '-output'
    if ext is not None:
        out_path += ext
    return out_path


parser = argparse.ArgumentParser(
    prog='landmark_video.py',
    description=("Estimates the pose of a single person in the given video file, "
                 "and outputs a video annotated with pose landmarks to the given "
                 "file path"))
parser.add_argument('input_file')
parser.add_argument('-o', '--output-file', dest='output_file')
parser.add_argument('-v', '--verbose',
                    choices=['true', 'false'], default='false', dest='verbose')
parser.add_argument("-f", "--fps", type=int, default=None, dest='fps',
                    help="FPS of output video")
parser.add_argument("-c", "--codec", type=str, default=None,
                    help="Codec of output video", dest='codec')
parser.add_argument("-W", "--width", dest='output_width', type=int, default=None,
                    help=("Width of output. Default is width of input. "
                          "Takes precedence over -s/--scale."))
parser.add_argument("-H", "--height", dest='output_height', type=int, default=None,
                    help=("Height of output. Default is height of input."
                          "Takes precedence over -s/--scale"))
parser.add_argument("-s", "--scale", dest='output_scale', type=int, default=100,
                    help=("Scale output by this many percent. Default is 100. "
                          "Has no effect if the above width & height values are set."))
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file or default_output_file_path(input_file)

# read the input video
cap = cv2.VideoCapture(input_file)
if cap.isOpened() is False:
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
output_fps = args.fps or int(cap.get(cv2.CAP_PROP_FPS))
output_width = args.output_width or int(frame_width * 0.01 * args.output_scale)
output_height = args.output_height or int(
    frame_height * 0.01 * args.output_scale)

# dimensions of the annotation panel
font_face = cv2.FONT_HERSHEY_PLAIN
font_scale = 0.3
color = (250,250,250)

# length of longest label + 2 spaces + 3 digits + degrees symbol
length_of_longest_label = max(len(k) for k,v in QuantifiedPose.ANGLE_LANDMARKS.items()) + 2 + 4

(label_width, label_height), baseline = cv2.getTextSize(("W" * length_of_longest_label), font_face, font_scale, 1)
panel_width = label_width + 4
panel_height = output_height

def decode_fourcc(four_cc_int_value):
    """FourCC values from a video file are packed into an int.
    We need to decode them into four actual characters if we
    want to re-use them.
    """
    return "".join([chr((int(four_cc_int_value) >> 8 * i) & 0xFF) for i in range(4)])

def print_debug_line(line):
    sys.stdout.write(str(line) + ' ')

output_codec = args.codec or decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))

# output video writer
out = cv2.VideoWriter(output_file,
                    cv2.VideoWriter_fourcc(*output_codec),
                    output_fps,
                    (output_width + panel_width, output_height))

if not out.isOpened():
    print("Error: Could not create the output video file.")
    cap.release()
    sys.exit()

if args.verbose == 'true':
    print('writing', cap.get(cv2.CAP_PROP_FRAME_COUNT), 'frames of annotated video to', output_file,
          'at', output_fps, 'fps,', output_width, 'x', output_height, 'with codec', output_codec)
    print('')

while cap.isOpened():
    # get frame
    ret, image = cap.read()
    if not ret:
        break

    # process the frame

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    if args.verbose == 'true':
        print_debug_line('Frame ' + str(cap.get(cv2.CAP_PROP_POS_FRAMES)) + \
                         ' of ' + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # if we didn't detect a pose, say so and skip
    if not results.pose_landmarks:
        if args.verbose == 'true':
            print_debug_line("No pose detected")
        continue
    else:
        quant_pose = QuantifiedPose(results.pose_world_landmarks, results.pose_landmarks)
        
        # calculate key body angles
        angles = quant_pose.calculate_angles()

        # draw the landmarks
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        

    # resize the frame if needed
    if output_width != frame_width or output_height != frame_height:
        # Resize the frame
        dim = (output_width, output_height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # add the annotation panel
    
    # Create a black image with extra width to hold both
    # max label size is 40ch (32ch label, 4 for value, 4 spaces for padding)
    # with 10px / char, that's 400px wide
    image_with_panel = np.zeros((output_height, output_width + panel_width, 3), np.uint8)
    
    # Copy in the annotated frame at top-left
    image_with_panel[0:output_height, 0:output_width] = image
    
    if results.pose_landmarks:
        # Write text for each angle
        top = 1
        
        for label, value in quant_pose.rounded_angles().items():
            # left-align the label
            cv2.putText(image_with_panel, 
                        label,
                        (output_width + 2, top),
                        font_face,
                        font_scale,
                        color,
                        1)
            # right-align the numerical value
            (value_width, value_height), baseline = cv2.getTextSize(str(value), font_face, font_scale, 1)
            cv2.putText(image_with_panel, 
                        str(value), 
                        ((output_width + panel_width + 2 - (value_width + baseline) - 1), top), 
                        font_face,
                        font_scale,
                        color,
                        1)
            top = top + value_height + 2

    # write the frame out
    out.write(image_with_panel)
    
    # wind the stdout buffer back a line if needed & flush
    if args.verbose == 'true':
        sys.stdout.write('\r\r')
        sys.stdout.flush()

# cleanup
pose.close()
cap.release()
out.release()

# print output file
print('\n')
print(output_file, ' - ', os.path.getsize(output_file), ' bytes')