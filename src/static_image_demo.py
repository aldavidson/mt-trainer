# cv2 - computer vision lib
# mediapip - google's toolkit for applying AI to media
import cv2
import mediapipe as mp

# shorthands for lib classes
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# read the image
input_file='/home/al/projects/aldavidson/mt-trainer/data/images/jom-kitti-front-teep-001.png'
mp_image = mp.Image.create_from_file(input_file)

# convert the image to the right format for pose recognition
converted_image = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_BGR2RGB)

# detect the pose
results = pose.process(converted_image)

# plot the pose as a connected skeleton in matlib3d
mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# Draw landmarks on the image itself
# Can only do this on a copy - the mp_image.numpy_view() is immutable
annotated_image = np.copy(mp_image.numpy_view())
converted_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
mp_drawing.draw_landmarks(converted_annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
# Save the annotated image
cv2.imwrite('/home/al/projects/aldavidson/mt-trainer/output/jk2.jpg', converted_annotated_image)