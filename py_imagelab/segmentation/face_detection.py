#!/usr/bin/env python
"""
Models

OpenCV Haar Cascade Frontal Face
Google's Mediapipe blaze_face_full_range.tflite

References
----------
.. [1] https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
.. [2] https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python
"""
import cv2
import logging
import mediapipe as mp
import numpy as np
import os
import pdb
from typing import Any
import urllib.request
from mediapipe.tasks import python as PythonMod
from mediapipe.tasks.python import vision as VisionMod
from py_imagelab.util import run_process, get_parser
import py_imagelab

logger = logging.getLogger(__name__)
class FaceDetectCascade():
    """Face Detection using the Haar Cascade model
    """
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier()

        # load face model from cv2 installation folder
        dir_data = cv2.__path__[0] + '/data/'
        model = "haarcascade_frontalface_default.xml"
        self.face_cascade.load(os.path.join(dir_data, model))
        logger.info(f"Loading OpenCV Face detection model {model}")

    def detect_face(self, im):
        if isinstance(im, str):
            im = cv2.imread(im)

        # NOTE: run detector on grayscale image
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.equalizeHist(im_gray)
        faces = self.face_cascade.detectMultiScale(im_gray)

        # NOTE: return color image, and detections
        return im, faces


class RobustFaceDetect():
    """Robust Face Detection handling extreme head tilts and angles
    using the Google MediaPipe BlazeFace machine learning model.
    """
    def __init__(self, min_detection_confidence=0.5):
        # Initialize MediaPipe Face Detection
        self.model_path = os.path.join(
            py_imagelab.__path__[0], "models/blaze_face_full_range.tflite")

        base_options = PythonMod.BaseOptions(model_asset_path=self.model_path)
        options = VisionMod.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence
        )
        self.detector = VisionMod.FaceDetector.create_from_options(options)

    def detect_face(self, im:str|np.ndarray[Any,Any]):
        if isinstance(im, str):
            im = cv2.imread(im)

        # NOTE: prepare image in format for MediaPipe
        #       Convert OpenCV BlueGreenRed format to RGB
        #       Then convert from numpy image to a MediaPipe Image.
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=im_rgb)
        results = self.detector.detect(mp_image)

        # initialize empty face detection lists
        faces = []
        frame_height, frame_width, _ = im.shape

        if results.detections:

            for detection in results.detections:
                # MediaPipe returns relative bounding boxes (0.0 to 1.0)
                bbox = detection.bounding_box
                x = bbox.origin_x
                y= bbox.origin_y
                width = bbox.width
                height = bbox.height

                # Append formatted coordinates to match your test_webcam loop structure
                faces.append((x, y, width, height))

                for key_point in detection.keypoints:
                    faces.append((
                        int(key_point.x * frame_width),
                        int(key_point.y * frame_height),
                        3, 3
                    ))

        return im, faces


if __name__ == "__main__":
    import py_imagelab
    py_imagelab.enable_color_logging()
    parser = get_parser()
    parser.add_argument("--mode", default="Haar", choices=["Haar", "Mediapipe"],)
    args = parser.parse_args()

    fdo : FaceDetectCascade |  RobustFaceDetect
    if args.mode == "Haar":
        fdo = FaceDetectCascade()
    elif args.mode == "Mediapipe":
        fdo = RobustFaceDetect()

    run_process(
        process=fdo.detect_face,
        params={},
        title="Face Detection",
        in_file=args.input,
        out_file=args.output,
        down=args.down,
        overwrite=args.overwrite,
        a_color=args.color
    )
