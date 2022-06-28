#!/usr/bin/env python
"""
References
----------
.. [1] https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
"""
import cv2
from py_imagelab.util import run_process, get_parser


class FaceDetectCascade():
    """Face Detection using the Haar Cascade model
    """
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier()

        # load face model from cv2 installation folder
        dir_data = cv2.__path__[0] + '/data/'
        self.face_cascade.load(dir_data + 'haarcascade_frontalface_default.xml')

    def detect_face(self, im):

        if isinstance(im, str):
            im = cv2.imread(im)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.equalizeHist(im_gray)

        faces = self.face_cascade.detectMultiScale(im_gray)
        return im, faces


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    fdo = FaceDetectCascade()


    run_process(
        process=fdo.detect_face,
        params={},
        title="Face Detection",
        in_file=args.input,
        out_file=args.output,
        down=args.down,
        overwrite=args.overwrite,
    )
