"""
This application performs an age/gender/emotion estimation based upon a
video source (videofile or webcam)

Usage:
      python demo_old.py

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
from utils.image import crop_bounding_box, draw_bounding_box_with_label, scale
import argparse
from utils.utils import str2bool, get_input_video_file_props
import os
import time
import sys
from my_keras_model import get_mobilenet_v2, get_opconty_shufflenet_v2


# ResNet sizing
DEPTH = 16
WIDTH = 8

# Face image size
FACE_SIZE = 64

# Model location
FACE_MODEL_PATH = 'pretrained_models/haarcascade_frontalface_alt.xml'
# FACE_MODEL_PATH = 'pretrained_models/haarcascade_frontalface_alt2.xml'
# FACE_MODEL_PATH = 'pretrained_models/lbpcascade_frontalface_improved.xml'
# FACE_MODEL_PATH = 'pretrained_models/haarcascade_frontalface_alt_tree.xml'
# FACE_MODEL_PATH = 'pretrained_models/haarcascade_frontalface_default.xml'

FACE_CAFFE_MODEL_TXT_PATH = 'pretrained_models/deploy.prototxt'
FACE_CAFFE_MODEL_PATH = 'pretrained_models/res10_300x300_ssd_iter_140000.caffemodel'

MY_MODEL_PATH = 'my_trained/21/weights.09-3.83.hdf5'

# saved dir for detected images
DET_EMOTION_DIR = 'output/emotion_images'


def get_gender_age_emotion(face_image):
    face_imgs = np.empty((1, FACE_SIZE, FACE_SIZE, 3))
    face_imgs[0, :, :, :] = face_image
    result = my_model.predict(face_imgs)
    output_gender = result[0][0]
    gender = "F" if output_gender[0] > 0.5 else "M"
    output_age = result[1][0]
    age = int(output_age.dot(np.arange(0, 101).reshape(101, 1)).flatten()[0])
    output_emotion = result[2][0]
    emotion_probability = np.max(output_emotion)
    emotion_label_arg = np.argmax(output_emotion)
    emotion = emotion_labels[emotion_label_arg]
    return gender, age, emotion


def make_dirs_for_emotion(video_file_name):
    for idx, emotion_label in emotion_labels.items():
        os.makedirs('{}/{}/{}'.format(DET_EMOTION_DIR, video_file_name, emotion_label), exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_webcam', type=str2bool, default=False, help='Specify if use webcam')
    parser.add_argument('--webcam_port', type=int, default=0, help='Specify webcam port')
    parser.add_argument('--input_video_path', type=str, default='input/dinner.mp4', help='Specify input video path')
    parser.add_argument('--output_emotion_images', type=str2bool, default=False,
                        help='Specify if output emotion images')
    parser.add_argument('--output_video', type=str2bool, default=False, help='Specify if output video')
    parser.add_argument('--slow_rate', type=float, default=1.0, help='Slower input fps')
    parser.add_argument('--face_confidence_threshold', type=float, default=0.2,
                        help='Specify confidence threshold for OpenCV DNN face detection')
    parser.add_argument('--use_cascade', type=str2bool, default=False,
                        help='Specify if use Cascade to do face detection')
    args = parser.parse_args()
    args.input_video_fname = os.path.basename(args.input_video_path)
    return args


def detect_face_with_opencv_haar(face_detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


def detect_face_with_opecv_dnn(face_detector, frame, threshold):
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and predictions
    face_detector.setInput(blob)
    detections = face_detector.forward()

    detected_faces = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < threshold:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        detected_faces.append((startX, startY, endX - startX, endY - startY))

    return detected_faces


def get_face_detector(use_cascade):
    if use_cascade:
        # HAAR Cascade model for face detection
        face_detector = cv2.CascadeClassifier(FACE_MODEL_PATH)
    else:
        # DNN model for face detection
        face_detector = cv2.dnn.readNetFromCaffe(FACE_CAFFE_MODEL_TXT_PATH,
                                                 FACE_CAFFE_MODEL_PATH)
    return face_detector


def detect_face(args, frame):
    if args.use_cascade:
        return detect_face_with_opencv_haar(face_detector, frame)
    else:
        return detect_face_with_opecv_dnn(face_detector, frame, args.face_confidence_threshold)


if __name__ == '__main__':
    args = get_args()
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad',
                      5: 'surprise', 6: 'neutral'}

    my_model = get_opconty_shufflenet_v2()
    # my_model = get_mobilenet_v2()
    # my_model = WideResNet(FACE_SIZE, depth=DEPTH, k=WIDTH)()
    my_model.load_weights(MY_MODEL_PATH)

    face_detector = get_face_detector(args.use_cascade)

    # Select video or webcam feed
    if args.use_webcam:
        capture = cv2.VideoCapture(args.webcam_port)
    else:
        if args.input_video_path:
            capture = cv2.VideoCapture(args.input_video_path)
            if capture.isOpened() is False:
                sys.exit('No video captured! Exit program~~~')
            input_video_props = get_input_video_file_props(capture)
            capture.set(cv2.CAP_PROP_FPS, input_video_props.fps * args.slow_rate)

    make_dirs_for_emotion(args.input_video_fname)

    frame_idx = 0
    detected_emotions = set()
    if args.output_video:
        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output/{}'.format(args.input_video_fname), fourcc,
                              input_video_props.fps * args.slow_rate,
                              (int(input_video_props.width), int(input_video_props.height)))

    while capture.isOpened():
        frame_start = time.time()
        success, frame = capture.read()
        if success is False:
            break

        frame_idx += 1

        # Detect faces
        face_detect_start = time.time()
        faces = detect_face(args, frame)
        face_detect_end = time.time()
        print('Detecting {} faces: {:.2f}ms'.format(len(faces), (face_detect_end - face_detect_start) * 1000))

        for face_idx, face in enumerate(faces):
            # Get face image, cropped to the size accepted by the WideResNet
            # face: (x, y, w, h)
            face_img, cropped, no_resized_img = crop_bounding_box(frame, face, margin=.4, size=(FACE_SIZE, FACE_SIZE))
            (x, y, w, h) = cropped

            # get gender, age, emotion
            predict_start = time.time()
            (gender, age, emotion) = get_gender_age_emotion(face_img)
            predict_end = time.time()
            print('Predicting gender, age, and emotion for one face: {:.2f}ms'.format(
                (predict_end - predict_start) * 1000))

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
