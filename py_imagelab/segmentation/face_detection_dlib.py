#!/usr/bin/env python
"""Face detection with
References
----------
.. [1] A. Rosebrock, "Face detection with dlib (HOG and CNN)", 2021,
    https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
.. [2] A. Ponnnusamy, "CNN based face detector from dlib"
    https://towardsdatascience.com/cnn-based-face-detector-from-dlib-c3696195e01c
"""
import cv2
import os
import dlib
from py_imagelab.util import run_process, get_parser


class FaceDetectDlib():
    def __init__(self, model=None):
        if model is not None and model != "":
            assert os.path.isfile(model), "Model file does not exist"
            self.detector = dlib.cnn_face_detection_model_v1(model)
        else:
            # default HOG
            self.detector = dlib.get_frontal_face_detector()
        self.upsample = 1

    def detect_face(self, im):
        if isinstance(im, str):
            im = cv2.imread(im)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.equalizeHist(im_gray)

        rects = self.detector(im_gray, self.upsample)
        faces = []
        if len(rects) > 0:
            for r in rects:
                faces.append([r.left(), r.top(), r.width(), r.height()])

        return im, faces


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--model", default="", help="if supplied will load model")
    args = parser.parse_args()

    fdo = FaceDetectDlib(args.model)


    run_process(
        process=fdo.detect_face,
        params={},
        title="Face Detection",
        in_file=args.input,
        out_file=args.output,
        down=args.down,
        overwrite=args.overwrite,
    )
