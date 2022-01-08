import mimetypes
import numpy as np
import os
import mimetypes
mimetypes.init()
from argparse import ArgumentParser
import cv2
from py_imagelab.test_with_webcam import test_webcam


def get_file_type(filepath):
    """Use mimetypes to guess file type

    Parameters
    ----------
    filepath : str
        The file in question

    Returns
    -------
    out : str or None
        File type from {'video', 'image'}.  Otherwise None
    """
    if not os.path.isfile(filepath):
        raise IOError("File (%s) does not exist" % filepath)

    guess, _ = mimetypes.guess_type(filepath)

    if "video" in guess:
        return "video"

    elif "image" in guess:
        return "image"

    else:
        return


def get_parser():
    """
    Get the basic argument parser with input, output, down, and overwrite

    Returns
    -------
    parser : ArgumentParser
        The base parser
    """
    parser = ArgumentParser()
    parser.add_argument("--input", default="", help="Input file")
    parser.add_argument("--output", default="/tmp/cartoon.png",
        help="Output image file")
    parser.add_argument("--down", default=1, type=int, help="Downsample image")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_process(process, params, title, in_file, out_file, down=1,
        overwrite=False):
    """Run process

    Parameters
    ----------
    process : function
        Function with the following signature
        (processed, detections) = process(in_image, **params)

    params : dict
        Dictionary of parameters specific to the process provided

    title : str
        The title to display in image.

    in_file : str
        The input file or directory, or "" == webcam.

    out_file : str
        The output file or directory

    down : int
        down sample with pyrDown.

    overwrite : bool
        Allow overwrite if true.
    """
    # -------------------------  check mode  --------------------------------
    if in_file:
        if os.path.isfile(in_file):
            mode = get_file_type(in_file)

        elif os.path.isdir(in_file):
            mode = "dir"

        else:
            raise IOError("Neither file nor dir")
    else:
        mode = "webcam"

    if mode != "dir" and not overwrite:
        # single file mode...check out_file
        assert not os.path.isfile(out_file), "File already exist...abort"

    # ----------------------------  run process  ----------------------------
    if mode == "image":
        # process single file
        # load file and downsample if provided
        image_rgb = cv2.imread(in_file)
        for _ in range(down):
            image_rgb = cv2.pyrDown(image_rgb)
        out_image, out_detect = process(image_rgb, **params)

        if out_detect is not None:
            for (x,y,w,h) in out_detect:
                thickness = int(1 + np.log10(np.min((w,h))))
                out_image = cv2.rectangle(out_image,
                    (x,y), (x+w, y+h), color=(200,0,0), thickness=thickness)
        cv2.imwrite(out_file, out_image)

    elif mode == "video":
        # process single video file
        test_webcam(out=out_file,
            process=process,
            params=params,
            title=title,
            cap=in_file,
            down=down)

    elif mode == "webcam":
        # process on webcam
        test_webcam(out=out_file,
            process=process,
            params=params,
            title=title,
            down=down)

    elif mode == "dir":
        # process file
        c_dir = os.path.abspath(in_file)

        # ----------------------  prepare output  ---------------------------
        o_file = os.path.basename(out_file)
        if len(o_file):
            o_dir = os.path.abspath(out_file[:-len(o_file)])
        else:
            o_dir = os.path.abspath(out_file)

        if not os.path.isdir(o_dir):
            os.makedirs(o_dir)

        # ------------------------  run process per file  -------------------
        files = os.listdir(in_file)
        for c_file in files:
            tmp_file = os.path.join(c_dir, c_file)
            c_mode = get_file_type(tmp_file)

            # set output file...check if it already exists
            out_file = os.path.join(o_dir, c_file)
            if os.path.isfile(out_file) and not overwrite:
                print("File (%s) exist...skipping" % out_file)

            # --------------  run process per image/video file  -------------
            if c_mode == "image":
                # load file and downsample if provided
                image_rgb = cv2.imread(tmp_file)
                for _ in range(down):
                    image_rgb = cv2.pyrDown(image_rgb)

                out_image, out_detect = process(image_rgb, **params)
                cv2.imwrite(out_file, out_image)

            elif c_mode == "video":
                test_webcam(out=out_file,
                    process=process,
                    params=params,
                    title=title,
                    cap=tmp_file,
                    down=down)
