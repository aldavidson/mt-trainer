import cv2
import mediapipe as mp
import numpy as np
import pdb
from PIL import Image, ImageDraw, ImageFont


from quantified_pose import QuantifiedPose
from text_rendering import Cv2TextRenderer


class FrameProcessor:
    '''
      Wrapper to encapsulate pose recognition and image annotation
    '''
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.pose_landmarker = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def release(self):
        self.pose_landmarker.close()

    def quantify_pose(self, rgb_image):
        '''
          Return a QuantifiedPose object encapsulating all that has been
          inferred about the pose of the main recognised person in the given
          rgb_image (if any)
        '''
        results = self.pose_landmarker.process(rgb_image)
        quant_pose = QuantifiedPose(results.pose_world_landmarks,
                                    results.pose_landmarks)
        return quant_pose

    def draw_landmarks(self, landmarks, rgb_image):
        '''
          Return a copy of the image with landmarks drawn & connected
          Can only do this on a copy - the mp_image.numpy_view() is immutable
        '''
        rgb_image_copy = np.copy(rgb_image)
        style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        mp.solutions.drawing_utils.draw_landmarks(
            rgb_image_copy,
            landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=style)

        return rgb_image_copy

    def dump_angles(self, font_size=12):
        ''' Make a new image with the given bg_color '''

    def render_angles(self,
                      quantified_pose,
                      image=None,
                      font_size=12,
                      font_face=cv2.FONT_HERSHEY_DUPLEX,
                      top=0,
                      left=0,
                      color=(255, 255, 255),
                      thickness=1,
                      text_renderer=None):

        renderer = text_renderer or Cv2TextRenderer()

        if image is None:
            image = self.make_panel_for_angles(font_size=font_size)

        label_top = top + font_size
        
        for label, value in quantified_pose.rounded_angles().items():
            renderer.render(label,
                            image,
                            top=label_top,
                            left=left,
                            pixel_height=font_size,
                            color=color,
                            thickness=thickness)

            image_width = len(image[0])
            # work out width of the value
            value_width = renderer.pixel_width(
                str(value),
                font_size,
                font_face=font_face,
                thickness=thickness
            )

            # render it at that many pixels from the right
            renderer.render(str(int(value)),
                            image,
                            top=label_top,
                            left=image_width - value_width,
                            pixel_height=font_size,
                            color=color,
                            thickness=thickness)

            label_top += font_size + 2

        return image

    def make_panel_for_angles(self,
                              font_size=12):
        ''' Make a panel just big enough to hold the body angles '''

        # height is (number of labels + 1 for frame no) * 
        # (height of label + space between each)
        height = (len(QuantifiedPose.ANGLE_LANDMARKS.keys()) + 1) * (font_size + 3)

        # width is font_size * 
        # (length of longest label + 2 chars space + 3 chars for angle)
        width = font_size * (QuantifiedPose.length_of_longest_label() + 5)
        image = np.zeros((height, width, 3), np.uint8)

        return image

    def append_image(self, image1, image2, padding=2):
        '''
          Append image2 to the top-right of image1.
          Leave (padding) pixels of space between them.
          Fill any blank space with zeros (black)
        '''
        image1_height, image1_width = (image1.shape[0], image1.shape[1])
        image2_height, image2_width = (image2.shape[0], image2.shape[1])
        combined_width = image1_width + image2_width + padding
        height = max(image1_height, image2_height)

        combined_image = np.zeros(
            (height, combined_width, 3),
            np.uint8
        )

        # copy the two images in
        combined_image[0:image1_height, 0:image1_width] = image1
        combined_image[0:image2_height, image1_width +
                       padding:combined_width] = image2

        return combined_image
