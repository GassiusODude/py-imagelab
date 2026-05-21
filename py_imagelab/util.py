#!/usr/bin/env python
import numpy as np
import os
import mimetypes
mimetypes.init()
import argparse
import cv2
import logging
import tempfile
from typing import Any, Callable
from py_imagelab.test_with_webcam import test_webcam

logger = logging.getLogger(__name__)

def get_file_type(filepath:str) -> str:
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
    if guess:
        if "video" in guess:
            return "video"

        elif "image" in guess:
            return "image"

    return ""

def rgb_int(value):
    ivalue = int(value)
    if ivalue < 0 or ivalue > 255:
        raise argparse.ArgumentTypeError(f"{value} is an invalid color channel. Must be between 0 and 255.")
    return ivalue

def get_parser() -> argparse.ArgumentParser:
    """
    Get the basic argument parser with input, output, down, and overwrite

    Returns
    -------
    parser : ArgumentParser
        The base parser
    """
    out_file = os.path.join(
        tempfile.gettempdir(),
        "image_out")
    logger.info(f"Default output file basename: {out_file}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", help="Input file")
    parser.add_argument("--output", default=out_file,
        help="Output image file")
    parser.add_argument("--down", default=1, type=int, help="Downsample image")
    parser.add_argument(
        "--color", nargs=3, type=rgb_int, default=(200,0,0) )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_process(
        process: Callable, params:dict, title:str, in_file:str, out_file:str,
        down:int=1, overwrite:bool=False, a_color:tuple[int,int,int]=(200,0,0)):
    """Run process

    Run the provided "process" with the associated "params".

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
        out_image, out_detect = _process_image(in_file, down, process, params, a_color)
        c_out_file = f"{out_file}.png"
        cv2.imwrite(c_out_file, out_image)
        logger.info(f"Wrote to {c_out_file}")

    elif mode == "video":
        # process single video file
        # FIXME: figure out the right way to process video file
        test_webcam(out=out_file + ".avi",
            process=process,
            params=params,
            title=title,
            cap=in_file,
            down=down)

    elif mode == "webcam":
        # process on webcam
        test_webcam(out=out_file + ".avi",
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
        # NOTE: loop through the input directory
        files = os.listdir(in_file)
        for c_file in files:
            tmp_file = os.path.join(c_dir, c_file)
            c_mode = get_file_type(tmp_file)

            # set output file...check if it already exists
            out_file = os.path.join(o_dir, c_file)
            if os.path.isfile(out_file) and not overwrite:
                logger.warning("File (%s) exist...skipping" % out_file)
                continue

            # --------------  run process per image/video file  -------------
            if c_mode == "image":
                # load file and downsample if provided
                out_image, out_detect = _process_image(tmp_file, down, process, params, a_color)
                # image_rgb = cv2.imread(tmp_file)
                # for _ in range(down):
                #     image_rgb = cv2.pyrDown(image_rgb)

                # out_image, out_detect = process(image_rgb, **params)
                cv2.imwrite(out_file, out_image)
                logger.info(f"Wrote to {out_file}")

            elif c_mode == "video":
                # FIXME: figure out the right way to process video file

                test_webcam(out=out_file,
                    process=process,
                    params=params,
                    title=title,
                    cap=tmp_file,
                    down=down)

def _process_image(
        in_file:str, down:int, process:Callable, params:dict,
        a_color:tuple[int,int,int]):
    """
    Parameters
    ----------
    in_file : str
        Input file path

    down : int
        The rounds of downsample

    process : Callable
        The process to apply to the image

    params : dict
        Parameters to the associated process

    a_color : tuple
        Annotation color for detections
    """
    # NOTE: read in input image and downsample
    image_rgb = cv2.imread(in_file)
    for _ in range(down):
        image_rgb = cv2.pyrDown(image_rgb)

    # NOTE: process the image
    out_image, out_detect = process(image_rgb, **params)

    # NOTE: if detections...draw rectangles to highlight
    if out_detect is not None:
        for (x, y, w, h) in out_detect:
            thickness = int(1 + np.log10(np.min((w,h))))
            out_image = cv2.rectangle(out_image,
                (x, y), (x + w, y + h), color=a_color, thickness=thickness)

    return out_image, out_detect