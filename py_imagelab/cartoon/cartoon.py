#!/usr/bin/python3
import numpy as np
import cv2

DEFAULT_CARTOONIFY = {
    "bilateral_stages": 7,
    "bilateral_diameter": 9,
    "bilateral_sigma_color": 9,
    "bilateral_sigma_space": 7,
    "blur_kernal_size":9,
    "adaptive_thresh_block": 9,
    "adaptive_thresh_const": 2,
}

def cartoonify(filename, out_size, out_file="/tmp/cartoon.png", **kwargs):
    """Cartoonify an input signal

    Reduces the number of color levels, remove detail and highlight
    certain edges.

    Notes
    -----
    According to the reference, use a bilateral filter to smooth the
    image and reduce the color scales.  Then apply edge detection on
    the grayscale of the image and adaptive detection before applying
    the enhanced edges on the blurred image.

    Parameters
    ----------
    filename : str
        The path to the input image

    out_size : tuple
        The output image size

    out_file : str
        The output filename

    References
    ----------
    .. [1] Available: https://www.askaswiss.com/2016/01/how-to-create-cartoon-effect-opencv-python.html#:~:text=Using%20OpenCV%20and%20Python%2C%20an,a%20cartoon%20in%20five%20steps%3A&text=Convert%20the%20original%20color%20image,grayscale%20image%20using%20adaptive%20thresholding.
    """

    # -------------  load from keyword arguments or use default  ------------
    n_bilat = kwargs.get("bilateral_stages", 7)
    bilat_diameter = kwargs.get("bilateral_diameter", 9)
    bilat_sigma_color = kwargs.get("bilateral_sigma_color", 9)
    bilat_sigma_space = kwargs.get("bilateral_sigma_space", 7)
    blur_kernal_size = kwargs.get("blur_kernal_size", 9)
    adapt_threshold_block = kwargs.get("adaptive_thresh_block", 9)
    adapt_threshold_const = kwargs.get("adaptive_thresh_const", 2)

    # ------------------  load image and downsample  ------------------------
    image_rgb = cv2.imread(filename)

    # identify the downsample
    down = np.max((int(image_rgb.shape[0] / out_size[0]),
        int(image_rgb.shape[1] / out_size[1])))

    # downsample by Gaussian pyramid
    for _ in range(down):
        image_rgb = cv2.pyrDown(image_rgb)

    # ------------------------  bilateral filter  ---------------------------
    for _ in range(n_bilat):
        image_rgb = cv2.bilateralFilter(image_rgb, d=bilat_diameter,
            sigmaColor=bilat_sigma_color, sigmaSpace=bilat_sigma_space)

    # -------------------------  median filter  -----------------------------
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, blur_kernal_size)

    # ------------------------  enhance edges  ------------------------------
    edges = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            blockSize=adapt_threshold_block, C=adapt_threshold_const)

    # convert back to color
    edges  = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # enchance edges
    cartoon_image = cv2.bitwise_and(image_rgb, edges)

    # -------------------------  save image  --------------------------------
    cv2.imwrite(out_file, cartoon_image)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("input", default="", help="Input file")
    parser.add_argument("--output", default="/tmp/cartoon.png",
        help="Output image file")
    parser.add_argument("--bi_stages", default=7, type=int,
        help="Number of stages of bilateral filter")
    parser.add_argument("--bi_diameter", default=9, type=int,
        help="Diameter of bilateral filter")
    parser.add_argument("--bi_sigma_color", default=9, type=int,
        help="Bilateral filter sigma color")
    parser.add_argument("--bi_sigma_space", default=7, type=int,
        help="Bilateral filter sigma space")
    parser.add_argument("--blur_kernal_size", default=9, type=int,
        help="Blur kernel size.")
    parser.add_argument("--adapt_thresh_block", default=9, type=int,
        help="Adaptive threshold block size")
    parser.add_argument("--adapt_thresh_const", default=2, type=int,
        help="Adaptive threshold constant")
    args = parser.parse_args()

    spec = {
        "bilateral_stages": args.bi_stages,
        "bilateral_diameter": args.bi_diameter,
        "bilateral_sigma_color": args.bi_sigma_color,
        "bilateral_sigma_space": args.bi_sigma_space,
        "blur_kernal_size":args.blur_kernal_size,
        "adaptive_thresh_block": args.adapt_thresh_block,
        "adaptive_thresh_const": args.adapt_thresh_const,
    }

    if args.input:
        cartoonify(args.input, (800, 600), out_file=args.output, **spec)