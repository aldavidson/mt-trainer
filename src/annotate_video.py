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
import math
import os
import sys
from time import time

import cv2
import mediapipe as mp
import numpy as np

from mt_trainer.frame_processor import FrameProcessor
from mt_trainer.text_rendering import Cv2TextRenderer
from mt_trainer.pose_classifier import PoseClassifier
from mt_trainer.graph_plotter import GraphPlotter
from mt_trainer.layout import Layout
from mt_trainer.camera import Camera
from mt_trainer import vector_maths

def default_output_file_path(path):
    """Append -output before the file extension in the given file path"""
    root, ext = os.path.splitext(path)
    out_path = root + '-output'
    if ext is not None:
        out_path += ext
    return out_path


def decode_fourcc(four_cc_int_value):
    """FourCC values from a video file are packed into an int.
    We need to decode them into four actual characters if we
    want to re-use them.
    """
    return "".join(
        [chr((int(four_cc_int_value) >> 8 * i) & 0xFF) for i in range(4)]
    )


def print_debug_line(*variables):
    ''' Writes the given line to STDOUT if in verbose mode, otherwise no-op '''
    if args.verbose == 'true':
        sys.stdout.write(' '.join([str(var) for var in variables]))


parser = argparse.ArgumentParser(
    prog='landmark_video.py',
    description=(
        "Estimates the pose of a single person in the given video file, "
        "and outputs a video annotated with pose landmarks to the given "
        "file path")
    )

parser.add_argument('input_file')
parser.add_argument('-o', '--output-file', dest='output_file')
parser.add_argument('-v', '--verbose',
                    choices=['true', 'false'], default='false', dest='verbose')
parser.add_argument('--from-frame', 
                    type=int, default=0, dest='from_frame',
                    help=("Start processing at this frame number"))
parser.add_argument('-m', '--max-frames',
                    type=int, default=None, dest='max_frames',
                    help=("Process a maximum of this many frames, starting at"
                          "--from-frame"))
parser.add_argument("-f", "--fps", type=int, default=None, dest='fps',
                    help="FPS of output video")
parser.add_argument("-c", "--codec", type=str, default=None,
                    help="Codec of output video", dest='codec')
parser.add_argument("-W", "--width", dest='output_width',
                    type=int, default=None,
                    help=("Width of video frames in output. "
                          "Default is width of input. "
                          "Takes precedence over -s/--scale."))
parser.add_argument("-H", "--height", dest='output_height',
                    type=int, default=None,
                    help=("Height of video frames in output."
                          "Default is height of input."
                          "Takes precedence over -s/--scale"))
parser.add_argument("-s", "--scale", dest='output_scale',
                    type=int, default=100,
                    help=("Scale output by this many percent. Default is 100. "
                          "Has no effect if the above width & height values"
                          "are set."))
parser.add_argument('--plot-3d', '-3d',
                    dest='plot_3d', default='false', choices=['false', 'true'])
parser.add_argument('-dc', '--min-detection-confidence',
                    dest='min_detection_confidence',
                    type=float, default=0.5)
parser.add_argument('-tc', '--min-tracking-confidence',
                    dest='min_tracking_confidence',
                    type=float, default=0.5)
parser.add_argument('-td', '--training-data',
                    dest='training_data_dir',
                    type=str, default='../data/poses/training/')

parser.add_argument('-cct', '--classification-confidence-threshold',
                    dest='classification_confidence_threshold',
                    type=float, default=0.98,
                    help=("Minimum confidence threshold for pose "
                          "classifications, on a scale from 0.0 to 1.0"
                          "0.98 is a good place to start."))
parser.add_argument('-ffc', '--frames-for-classification',
                    dest='frames_for_classification',
                    type=int, default=3,
                    help=("Minimum number of successive frames for which a" 
                          "pose classification must persist in order to be"
                          "outputted. Larger values help make the pose"
                          "classification less noisy"))

args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file or default_output_file_path(input_file)

classifier = PoseClassifier(data_dir=args.training_data_dir)

# read the input video
cap = cv2.VideoCapture(input_file)
if cap.isOpened() is False:
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
max_frames = args.max_frames or (int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - args.from_frame))
output_fps = args.fps or int(cap.get(cv2.CAP_PROP_FPS))
output_frame_width = args.output_width or int(frame_width * 0.01 * args.output_scale)
output_frame_height = args.output_height or int(
    frame_height * 0.01 * args.output_scale)

processor = FrameProcessor(
    min_detection_confidence=args.min_detection_confidence,
    min_tracking_confidence=args.min_tracking_confidence)

text_renderer = Cv2TextRenderer()

# need to do this now, so that we can work out the output width for the video
# 
# layout:
#
# ---------------------------------------------
# | original video, scaled | body angles & prediction |
# height is adjusted to the tallest of the above
# width also includes a few pixels padding between the two panels
# 
# if told to plot3d, we append another row on the bottom:
# | 3d landmarks           | (empty space)            |
# -----------------------------------------------------
FONT_SIZE = 12
annotation_panel = processor.make_panel_for_angles(font_size=FONT_SIZE)
PADDING = 2
# panel_width = annotation_panel.shape[1]
# height_with_annotation = max(output_frame_height, annotation_panel.shape[0])
# annotated_video_width = output_frame_width + panel_width + PADDING

if args.plot_3d == 'true':
    panel_3d_size = [output_frame_width, output_frame_height]
else:
    panel_3d_size = None


layout = Layout(
    [output_frame_width, output_frame_height],
    [annotation_panel.shape[1], annotation_panel.shape[0]],
    panel_3d_size,
    PADDING,
)

output_codec = args.codec or decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))

# output video writer
out = cv2.VideoWriter(output_file,
                      cv2.VideoWriter_fourcc(*output_codec),
                      output_fps,
                      (layout.total_width, layout.total_height))

if not out.isOpened():
    print("Error: Could not create the output video file.")
    cap.release()
    sys.exit()
    
# Create the graph here as it's an expensive operation
plotter = None
if args.plot_3d == 'true':
    plotter = GraphPlotter()
    camera = Camera(image_width=panel_3d_size[0],
                    image_height=panel_3d_size[1],
                    )
    # rotate the camera at PI/4 radians/sec (=> 8s per full circle)
    # angular_speed = 0.25 * math.pi
    angular_speed = 0.0

print_debug_line('writing', max_frames, 
                 'frames of annotated video to', output_file,
                 'at', output_fps, 'fps,', 
                 layout.video_size[0], 'x', layout.video_size[1],
                 'with codec', output_codec,
                 ' total size =',
                 layout.total_width, 'x', layout.total_height)
print_debug_line('\n\n')

output_frame_number = 1
last_classification = None
frames_with_this_classification = 0
last_3d_render_time = None

while (cap.isOpened() and 
       (cap.get(cv2.CAP_PROP_POS_FRAMES) <= (args.from_frame + max_frames))):
    start = time()

    # get frame
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ret, input_image = cap.read()
    if not ret:
        print("Couldn't read frame ", frame_number, "from", input_file,
              "aborting!")
        break

    # Skip if < from_frame
    if int(frame_number) < args.from_frame:
        print_debug_line('Skipping frame ',  int(frame_number))
        sys.stdout.write('\r')
        sys.stdout.flush()
        # ignore skip time in calculations of FPS
        start = time()
        whole_process_start = time()
        continue

    print_debug_line('Frame ', frame_number,
                     ' of ', cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # process the frame
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image.flags.writeable = True

    pose = processor.quantify_pose(input_image)

    panel = processor.make_panel_for_angles(FONT_SIZE)

    # if we didn't detect a pose, say so and skip
    if not pose:
        # we'll just output the input image
        output_image = input_image

        print_debug_line(" No pose detected")
        sys.stdout.write('\r')
        sys.stdout.flush()
        continue
    else:
        # draw the landmarks
        output_image = processor.draw_landmarks(
            pose.image_landmarks,
            input_image
        )

        # render the frame number into the panel
        text_renderer.render('Frame #' + str(int(output_frame_number)),
                             panel,
                             top=FONT_SIZE+2, left=2,
                             pixel_height=FONT_SIZE,
                             color=(255, 255, 255))

        # render the body angles into the panel
        panel = processor.render_angles(
            pose, panel, top=FONT_SIZE * 2, font_size=FONT_SIZE)
        
        # render the pose classification
        classification = classifier.classify(
            pose,
            threshold=args.classification_confidence_threshold,
            max_results=1
        )
        # we get an array back, it might be empty
        # But if it isn't ....
        if classification:
            print_debug_line(classification[0])
            confidence = classification[0][1]
            classification = classification[0][0]
            if classification == last_classification:
                frames_with_this_classification += 1
            else:
                frames_with_this_classification = 1
                
            # only output the classification if it's been constant for
            # at least the required number of frames
            if frames_with_this_classification >= args.frames_for_classification:    
                top = panel.shape[0] - (FONT_SIZE + 2)
                confidence_pct = str(round(100.0 * confidence, 2))
                prediction = f"Pose: {classification} ({confidence_pct}%)"
                text_renderer.render(
                    prediction,
                    panel,
                    top=top, left=2,
                    pixel_height=FONT_SIZE,
                    color=(255, 255, 255))
            
            last_classification = classification
        else:
            print_debug_line(
                "doesn't match any known pose by at least",
                f"{round(args.classification_confidence_threshold, 2)}%"
            )
            
    # resize the frame if needed
    if output_frame_width != frame_width or output_frame_height != frame_height:
        # Resize the frame
        dim = (output_frame_width, output_frame_height)
        output_image = cv2.resize(
            output_image, dim, interpolation=cv2.INTER_AREA)

    # combine the landmarked image and annotation panel into one
    output_image_with_panel = processor.append_image_to_rhs(output_image, panel)
    
    # plot the pose as a connected skeleton in matlib3d if required
    if args.plot_3d == 'true':
        image_3d = np.zeros((layout.video_size[1],
                             layout.video_size[0],
                             3),
                            np.uint8
                            )
        # white background
        image_3d.fill(255)
        
        start = time()
        # rotate the camera around the y axis
        if False and last_3d_render_time:
            rotation_angle = (angular_speed * (time() - last_3d_render_time))
            camera.rotation_vector[2] = 2
            camera.rotation_vector[1] = camera.rotation_vector[1] + rotation_angle
            print_debug_line( 'rotation_angle is ', rotation_angle )
            print_debug_line( 'Camera position is ', camera.position )
            
        plotter.plot_3d_landmarks_on_image(landmark_list=pose.world_landmarks,
                                           image=image_3d,
                                           camera=camera)
        print_debug_line(' Plotted 3d landmarks in ', str(round(time() - start, 4)) + 's')
        
        last_3d_render_time = time()
        
        start = time()
        output_image_with_panel = processor.append_image_to_bottom_left(
            output_image_with_panel,
            image_3d)
        print_debug_line(' appended image in ', str(round(time() - start, 4)) + 's')

    # write the frame out
    out.write(cv2.cvtColor(output_image_with_panel, cv2.COLOR_RGB2BGR))
    output_frame_number += 1

    print_debug_line(' - Total frame time', str(round(time() - start, 4)) + 's')

    # wind the stdout buffer back a line if needed & flush
    if args.verbose == 'true':
        sys.stdout.write('\r')
        sys.stdout.flush()

whole_process_time = time() - whole_process_start
print_debug_line('\nProcessed', output_frame_number, 
                 'frames in ', str(round(whole_process_time, 2)) + 's',
                 '=>', round(output_frame_number / whole_process_time, 2), 'fps')
# cleanup
processor.pose_landmarker.close()
if plotter:
    plotter.cleanup()
cap.release()
out.release()

# print output file
print('\n')
print(output_file, ' - ', os.path.getsize(output_file), ' bytes')
