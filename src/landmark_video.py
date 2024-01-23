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

from PIL import Image, ImageDraw, ImageFont

import cv2
import mediapipe as mp
import numpy as np
from quantified_pose import QuantifiedPose
from text_rendering import Cv2TextRenderer, PILTextRenderer

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
font_face = "SourceSans3-Regular.ttf" # cv2.FONT_HERSHEY_PLAIN
font_height = 12
font_scale = 0.3
color = (250,250,250)

text_renderer = PILTextRenderer("SourceSans3-Regular.ttf", "./assets/")
length_of_longest_label = max(len(k) for k,v in QuantifiedPose.ANGLE_LANDMARKS.items()) + 2 + 4

# length of longest label + 2 spaces + 3 digits + degrees symbol
label_width = length_of_longest_label * 14

# (label_width, label_height), baseline = cv2.getTextSize(("W" * length_of_longest_label), font_face, font_scale, 1)

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

# load the nice TTF font for use with pillow
nice_font = ImageFont.truetype("./assets/SourceSans3-Regular.ttf", 10) 
# # # 
def draw_text_with_pillow(image, text, origin, color='#FFF'):
#     # convert color format to PIL-compatible
#     cv2_im_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#      # Pass the image to PIL  
#     pil_im = Image.fromarray(cv2_im_rgb)
    # draw the text
    draw = ImageDraw.Draw(image) #pil_im
    draw.text(origin, text, font=nice_font, fill=color)
    # convert it back to OpenCV format
    # return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

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
    ret, input_image = cap.read()
    if not ret:
        break

    if args.verbose == 'true':
        print_debug_line('Frame ' + str(cap.get(cv2.CAP_PROP_POS_FRAMES)) + \
                         ' of ' + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # process the frame
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image.flags.writeable = True
    
    results = pose.process(input_image)

    # resize the frame if needed
    if output_width != frame_width or output_height != frame_height:
        # Resize the frame
        dim = (output_width, output_height)
        input_image = cv2.resize(input_image, dim, interpolation=cv2.INTER_AREA)
    
    # convert it to cv2 BGR format
    cv2_image_with_landmarks = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        
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
        mp_drawing.draw_landmarks(
            cv2_image_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        

        # Create a black image with extra width to hold input_image + annotation panel
        # max label size is 40ch (32ch label, 4 for value, 4 spaces for padding)
        # with 10px / char, that's 400px wide
        output_image_with_panel = np.zeros((output_height, output_width+panel_width, 3), np.uint8)
        output_image_with_panel = cv2.cvtColor( output_image_with_panel, cv2.COLOR_RGB2BGR)
        
        annotation_panel = np.zeros((output_height, panel_width, 3), np.uint8)
        # convert color format to PIL-compatible
        cv2_im_rgb = cv2.cvtColor(annotation_panel,cv2.COLOR_BGR2RGB)
         # Pass the image to PIL  
        pil_annotation_panel = Image.fromarray(cv2_im_rgb)
        
        # Write text for each angle
        top = 10
        
        for label, value in quant_pose.rounded_angles().items():
            # left-align the label
            # cv2.putText(annotation_panel, 
            #             label,
            #             (2, top),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             font_scale,
            #             color,
            #             1)
            # pil_annotation_panel = draw_text_with_pillow(pil_annotation_panel, label, (output_width + 2, top))
            
            # pil_image_with_panel = text_renderer.render(
            #     label,
            #     pil_image_with_panel,
            #     top=top,
            #     left=(output_width + 2),
            #     pixel_height=font_height,
            #     color="#FFF",
            #     thickness=1,
            # )
            
            pil_annotation_panel = text_renderer.render(
                label,
                pil_annotation_panel,
                top=top,
                left=(output_width + 2),
                color="#FFF",
                font=nice_font,
            )
            
            # right-align the numerical value
            value_string = str(int(value)) + '°'
            (value_width, value_height), baseline = cv2.getTextSize(value_string, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            # value_width = text_renderer.pixel_width(
            #     str(value),
            #     pil_annotation_panel,
            #     font_height,
            #     font_face=font_face)
            
            # cv2.putText(annotation_panel, 
            #             value_string, 
            #             ((panel_width - value_width - 1), top), 
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             font_scale,
            #             color,
            #             1)
            
            # pil_annotation_panel = draw_text_with_pillow(pil_annotation_panel, str(value), ((output_width + panel_width + 2 - (value_width) - 1), top))
            
            pil_annotation_panel = text_renderer.render(
                str(value),
                pil_annotation_panel,
                top=top,
                left=(panel_width - value_width - 2),
                align='right',
                pixel_height=10,
                color="#FFF",
                thickness=1,
                font=nice_font,
            )
            
            top = top + 14

        # convert it back to OpenCV format
        cv2_annotation_panel = cv2.cvtColor(np.array(pil_annotation_panel), cv2.COLOR_RGB2BGR)
        
        # and copy it into the output image 
        output_image_with_panel[0:output_height, output_width:(output_width+panel_width)] = cv2_annotation_panel
        # output_image_with_panel[0:output_height, output_width:(output_width+panel_width)] = annotation_panel
            
    # Copy in the annotated frame at top-left
    output_image_with_panel[0:output_height, 0:output_width] = cv2_image_with_landmarks
    
    # write the frame out
    out.write(output_image_with_panel)
    
    
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