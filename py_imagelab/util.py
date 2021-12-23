import mimetypes
import os
import mimetypes
mimetypes.init()
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


def run_process(process, params, title, in_file, out_file):
    
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

    if mode != "dir":
        # single file mode...check out_file
        assert not os.path.isfile(out_file), "File already exist...abort"

    # ----------------------------  run process  ----------------------------
    if mode == "image":
        # process single file
        out_image, out_detect = process(in_file, **params)
        cv2.imwrite(out_file, out_image)

    elif mode == "video":
        # process single video file
        test_webcam(out=out_file,
            process=process,
            params=params,
            title=title,
            cap=in_file)

    elif mode == "webcam":
        # process on webcam
        test_webcam(out=out_file,
            process=process,
            params=params,
            title=title)

    elif mode == "dir":
        # process file
        c_dir = os.path.abspath(in_file)
        #o_dir = os.path.dirname(os.path.abspath(out_file))
        o_file = os.path.basename(out_file)
        if len(o_file):
            o_dir = os.path.abspath(out_file[:-len(o_file)])
        else:
            o_dir = os.path.abspath(out_file)
        if not os.path.isdir(o_dir):
            os.makedirs(o_dir)
        files = os.listdir(in_file)
        for c_file in files:
            tmp_file = os.path.join(c_dir, c_file)
            c_mode = get_file_type(tmp_file)

            # set output file...check if it already exists
            out_file = os.path.join(o_dir, c_file)
            if os.path.isfile(out_file):
                print("File (%s) exist...skipping" % out_file)

            # --------------  run process per image/video file  -------------
            if c_mode == "image":
                out_image, out_detect = process(tmp_file, **params)
                cv2.imwrite(out_file, out_image)

            elif c_mode == "video":
                test_webcam(out=out_file,
                    process=process,
                    params=params,
                    title=title,
                    cap=tmp_file)
