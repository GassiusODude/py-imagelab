"""
References
----------
.. [1] https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
"""
import cv2

class FaceDetectCascade():
    """
    """
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier()

        # load face model from cv2 installation folder
        dir_data = cv2.__path__[0] + '/data/'
        self.face_cascade.load(dir_data + 'haarcascade_frontalface_default.xml')
        #self.face_cascade.load(dir_data + 'haarcascade_frontalface_alt_tree.xml')
        

    def detect_face(self, im):
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.equalizeHist(im_gray)
        
        faces = self.face_cascade.detectMultiScale(im_gray)
        return im, faces

if __name__ == "__main__":
    from py_imagelab.test_with_webcam import test_webcam
    fdo = FaceDetectCascade()
    test_webcam(process=fdo.detect_face, title="Face Detection")