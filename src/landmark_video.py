#!/usr/bin/python

# cv2 - computer vision lib
# mediapipe - google's toolkit for applying AI to media
import argparse
import cv2
import mediapipe as mp
import numpy as np
import os

# shorthands for lib classes
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def default_output_file_path(input_file):
  root, ext = os.path.splitext(input_file)
  out_path = root + '-output'
  if ext != None:
    out_path += ext
  return out_path

parser = argparse.ArgumentParser(
                    prog='landmark_video.py',
                    description='Estimates the pose of a single person in the given video file, outputs an annotated video to the given file path')
parser.add_argument('input_file')
parser.add_argument('--output-file', '-o', dest='output_file')
parser.add_argument('-v', '--verbose', choices=['true','false'], default='false', dest='verbose')
parser.add_argument("-f", "--fps", type=int, default=None, dest='fps',
	help="FPS of output video")
parser.add_argument("-c", "--codec", type=str, default=None,
	help="Codec of output video", dest='codec')
args = parser.parse_args()
input_file=args.input_file
output_file=args.output_file or default_output_file_path(input_file)

# read the input video
cap = cv2.VideoCapture(input_file)
if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
output_fps = args.fps or int(cap.get(cv2.CAP_PROP_FPS))

def decode_fourcc(cc):
  return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])

output_codec=args.codec or decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
print('codec = ', output_codec)

# output video writer
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(
    *output_codec), output_fps, (frame_width, frame_height))

if not out.isOpened():
    print("Error: Could not create the output video file.")
    cap.release()
    exit()

print('writing ', cap.get(cv2.CAP_PROP_FRAME_COUNT), ' frames of annotated video to ', output_file, ' at ', output_fps, 'fps, ', frame_width, 'x', frame_height, ' with codec ', output_codec)

while cap.isOpened():
    # get frame
    ret, image = cap.read()
    if not ret:
        break

    # process the frame
    if args.verbose == 'true':
      print('Frame ', cap.get(cv2.CAP_PROP_POS_FRAMES), ' of ', cap.get(cv2.CAP_PROP_FRAME_COUNT))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    # draw the landmarks
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # write the frame out
    out.write(image)

# cleanup
pose.close()
cap.release()
out.release()
