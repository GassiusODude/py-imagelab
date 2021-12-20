import cv2
import numpy as np
class Foreground():
    def __init__(self,):
        self.mask = None

        self.bg_model = np.zeros((1, 65), np.float64)
        self.fg_model = np.zeros((1, 65), np.float64)

    def extract_foreground(self, img, rect=None):
        """GrabCut version to extract foreground

        GrabCut uses an initial bounding box of the foreground.  An 
        iterative approach ensues with GMM modeling the foreground
        and background.
        """
        if self.mask is None:
            self.mask = np.zeros(img.shape[:2], np.uint8)

        # initial detection box
        if rect is None:
            width = img.shape[1]
            height = img.shape[0]
            rect = [int(0.25 * width), int(0.25*height), int(0.5*width), int(0.5*height)]
        
        cv2.grabCut(img, self.mask, rect, self.bg_model, self.fg_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((self.mask==2)|(self.mask==0), 0,1).astype('uint8')

        img_out = img * mask2[:,:, np.newaxis]
        return img_out, None


if __name__ == "__main__":
    from py_imagelab.test_with_webcam import test_webcam
    fc = Foreground()
    test_webcam(process=fc.extract_foreground, title="GrabCut")