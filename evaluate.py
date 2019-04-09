"""
This application performs an age/gender/emotion estimation based upon a
video source (videofile or webcam)

Usage:
      python evaluate.py

For now, switching input source is done with global variables in the source code

The following work is used:
1. OpenCV2 haar cascade face recognition from https://github.com/opencv/opencv/
2. Gender and Age model                  from https://github.com/Tony607/Keras_age_gender
3. Emotion model                         from https://github.com/petercunha/Emotion
4. Wide Resnet implementation            from https://github.com/asmith26/wide_resnets_keras
"""
import numpy as np
from keras.models import load_model
import cv2
from wide_resnet import WideResNet
from utils.array import scale
from utils.image import crop_bounding_box, draw_bounding_box_with_label
import argparse
from utils.utils import str2bool, get_input_video_file_props
import os
import re
import time
import sys

# ResNet sizing
DEPTH = 16
WIDTH = 8

# Face image size
FACE_SIZE = 64

# Model location
# Face model from CV2 (Haar cascade)
FACE_MODEL_PATH = 'models/haarcascade_frontalface_alt.xml'
# Gender and Age model from
# https://github.com/Tony607/Keras_age_gender (weights.18-4.06.hdf5)
# https://github.com/yu4u/age-gender-estimation (weights.28-3.73.hdf5, weights.29-3.76_utk.hdf5)
# with Wide ResNet from https://github.com/asmith26/wide_resnets_keras
AGENDER_MODEL_PATH = 'models/weights.28-3.73.hdf5'
# Emotion model from https://github.com/petercunha/Emotion
EMOTION_MODEL_PATH = 'models/emotion_model.hdf5'

# saved dir for detected images
DET_EMOTION_DIR = 'output/emotion_images'


def get_age_gender(face_image):
    """
    Determine the age and gender of the face in the picture
    :param face_image: image of the face
    :return: (age, gender) of the image
    """
    face_imgs = np.empty((1, FACE_SIZE, FACE_SIZE, 3))
    face_imgs[0, :, :, :] = face_image
    result = agender_model.predict(face_imgs)
    est_gender = "F" if result[0][0][0] > 0.5 else "M"
    est_age = int(result[1][0].dot(np.arange(0, 101).reshape(101, 1)).flatten()[0])
    return est_age, est_gender


def get_emotion(face_image):
    """
    Determine the age and gender of the face in the picture
    :param face_image: image of the face
    :return: str:emotion of the image
    """
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray_face = scale(gray_face)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)

    # Get EMOTION
    emotion_prediction = emotion_model.predict(gray_face)
    emotion_probability = np.max(emotion_prediction)
    emotion_label_arg = np.argmax(emotion_prediction)
    return emotion_labels[emotion_label_arg]


def make_dirs_for_emotion(video_file_name):
    for idx, emotion_label in emotion_labels.items():
        os.makedirs('{}/{}/{}'.format(DET_EMOTION_DIR, video_file_name, emotion_label), exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_webcam', type=str2bool, default=True)
    parser.add_argument('--webcam_port', type=int, default=0)
    parser.add_argument('--input_video_path', type=str, default='input/dinner.mp4')
    parser.add_argument('--output_emotion_images', type=str2bool, default=False)
    parser.add_argument('--output_video', type=str2bool, default=False)
    args = parser.parse_args()
    regex = re.compile(r'input\/(.+)')
    match = regex.search(args.input_video_path)
    args.input_video_fname = match.group(1)
    return args


if __name__ == '__main__':
    args = get_args()

    # WideResNet model for Age and Gender
    agender_model = WideResNet(FACE_SIZE, depth=DEPTH, k=WIDTH)()
    agender_model.load_weights(AGENDER_MODEL_PATH)

    # VCC model for emotions
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad',
                      5: 'surprise', 6: 'neutral'}
    emotion_model = load_model(EMOTION_MODEL_PATH)
    emotion_target_size = emotion_model.input_shape[1:3]

    # Cascade model for face detection
    face_cascade = cv2.CascadeClassifier(FACE_MODEL_PATH)

    # Select video or webcam feed
    if args.use_webcam:
        capture = cv2.VideoCapture(args.webcam_port)
    else:
        if args.input_video_path:
            capture = cv2.VideoCapture(args.input_video_path)
            if capture.isOpened() is False:
                sys.exit('No video captured! Exit program~~~')
            input_video_props = get_input_video_file_props(capture)

    make_dirs_for_emotion(args.input_video_fname)

    frame_idx = 0
    detected_emotions = set()
    if args.output_video:
        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'divx')
        out = cv2.VideoWriter('output/{}'.format(args.input_video_fname), fourcc,
                              input_video_props.fps,
                              (int(input_video_props.width), int(input_video_props.height)))

    while capture.isOpened():
        frame_start = time.time()
        success, frame = capture.read()
        if success is False:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_detect_start = time.time()
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_detect_end = time.time()
        print('Detecting {} faces: {:.2f}s'.format(len(faces), face_detect_end - face_detect_start))

        for face_idx, face in enumerate(faces):
            # Get face image, cropped to the size accepted by the WideResNet
            face_img, cropped, no_resized_img = crop_bounding_box(frame, face, margin=.4, size=(FACE_SIZE, FACE_SIZE))
            (x, y, w, h) = cropped

            # Get AGE and GENDER and EMOTION
            agender_detect_start = time.time()
            (age, gender) = get_age_gender(face_img)
            agender_detect_end = time.time()
            print('Predicting age and gender for one face: {:.2f}s'.format(agender_detect_end - agender_detect_start))
            emotion_detect_start = time.time()
            emotion = get_emotion(face_img)
            emotion_detect_end = time.time()
            print('Predicting emotion for one face: {:.2f}s'.format(emotion_detect_end - emotion_detect_start))

            # save image for each emotion
            detected_emotions.add(emotion)
            if args.output_emotion_images:
                cv2.imwrite(
                    '{:s}/{:s}/{:s}/{:s}_{:0>4d}_{:0>2d}.png'.format(DET_EMOTION_DIR, args.input_video_fname, emotion,
                                                                     emotion, frame_idx, face_idx),
                    no_resized_img)

            # Add box and label to image
            label = "{}, {}, {}".format(age, gender, emotion)
            draw_bounding_box_with_label(frame, x, y, w, h, label)

        frame_end = time.time()
        fps = 1 / (frame_end - frame_start)
        cv2.putText(frame, 'FPS: {:.1f}'.format(fps),
                    (int(input_video_props.width - 125), int(input_video_props.height - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0), 2)

        # Display the resulting image
        cv2.imshow('Video', frame)

        if args.output_video:
            out.write(frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    capture.release()
    if args.output_video:
        out.release()
    cv2.destroyAllWindows()

    print('detected emotions: {}'.format(detected_emotions))
