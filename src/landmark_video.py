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


import cv2
import mediapipe as mp
import numpy as np

from frame_processor import FrameProcessor

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
    description=(
        "Estimates the pose of a single person in the given video file, "
        "and outputs a video annotated with pose landmarks to the given "
        "file path"))
parser.add_argument('input_file')
parser.add_argument('-o', '--output-file', dest='output_file')
parser.add_argument('-v', '--verbose',
                    choices=['true', 'false'], default='false', dest='verbose')
parser.add_argument('-m', '--max-frames',
                    type=int, default=None, dest='max_frames')
parser.add_argument("-f", "--fps", type=int, default=None, dest='fps',
                    help="FPS of output video")
parser.add_argument("-c", "--codec", type=str, default=None,
                    help="Codec of output video", dest='codec')
parser.add_argument("-W", "--width", dest='output_width',
                    type=int, default=None,
                    help=("Width of output. Default is width of input. "
                          "Takes precedence over -s/--scale."))
parser.add_argument("-H", "--height", dest='output_height',
                    type=int, default=None,
                    help=("Height of output. Default is height of input."
                          "Takes precedence over -s/--scale"))
parser.add_argument("-s", "--scale", dest='output_scale',
                    type=int, default=100,
                    help=("Scale output by this many percent. Default is 100. "
                          "Has no effect if the above width & height values"
                          "are set."))
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
max_frames = args.max_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
output_fps = args.fps or int(cap.get(cv2.CAP_PROP_FPS))
output_width = args.output_width or int(frame_width * 0.01 * args.output_scale)
output_height = args.output_height or int(
    frame_height * 0.01 * args.output_scale)


def decode_fourcc(four_cc_int_value):
    """FourCC values from a video file are packed into an int.
    We need to decode them into four actual characters if we
    want to re-use them.
    """
    return "".join(
        [chr((int(four_cc_int_value) >> 8 * i) & 0xFF) for i in range(4)]
    )


def print_debug_line(line):
    sys.stdout.write(str(line) + ' ')


processor = FrameProcessor()
# need to do this now, so that we can work out the output width for the video
FONT_SIZE = 12
annotation_panel = processor.make_panel_for_angles(font_size=FONT_SIZE)
PADDING = 2
panel_width = annotation_panel.shape[1]
height_with_annotation = max(output_height, annotation_panel.shape[0])
annotated_video_width = output_width + panel_width + PADDING

output_codec = args.codec or decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))

# output video writer
out = cv2.VideoWriter(output_file,
                      cv2.VideoWriter_fourcc(*output_codec),
                      output_fps,
                      (annotated_video_width, height_with_annotation))

if not out.isOpened():
    print("Error: Could not create the output video file.")
    cap.release()
    sys.exit()

if args.verbose == 'true':
    print('writing', max_frames, 'frames of annotated video to', output_file,
          'at', output_fps, 'fps,', output_width, 'x', output_height,
          'with codec', output_codec, ' shape with panel =',
          (annotated_video_width, output_height))
    print('')


while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) <= max_frames:
    # get frame
    ret, input_image = cap.read()
    if not ret:
        break

    if args.verbose == 'true':
        print_debug_line('Frame ' + str(cap.get(cv2.CAP_PROP_POS_FRAMES)) +
                         ' of ' + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # process the frame
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image.flags.writeable = True

    pose = processor.quantify_pose(input_image)

    panel = processor.make_panel_for_angles()

    # if we didn't detect a pose, say so and skip
    if not pose.world_landmarks:
        # we'll just output the input image
        output_image = input_image

        if args.verbose == 'true':
            print_debug_line("No pose detected")
        continue
    else:
        # draw the landmarks
        output_image = processor.draw_landmarks(
            pose.image_landmarks,
            input_image
        )

        # render the body angles into the panel
        panel = processor.render_angles(pose, panel, font_size=FONT_SIZE)

    # resize the frame if needed
    if output_width != frame_width or output_height != frame_height:
        # Resize the frame
        dim = (output_width, output_height)
        output_image = cv2.resize(
            output_image, dim, interpolation=cv2.INTER_AREA)

    # combine the landmarked image and annotation panel into one
    output_image_with_panel = processor.append_image(output_image, panel)

    # write the frame out
    out.write(cv2.cvtColor(output_image_with_panel, cv2.COLOR_RGB2BGR))

    # wind the stdout buffer back a line if needed & flush
    if args.verbose == 'true':
        sys.stdout.write('\r')
        sys.stdout.flush()

# cleanup
processor.pose_landmarker.close()
cap.release()
out.release()

# print output file
print('\n')
print(output_file, ' - ', os.path.getsize(output_file), ' bytes')
