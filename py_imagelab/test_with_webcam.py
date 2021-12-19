import cv2

def test_webcam(out=None, process=None, params=None, title="Preview"):
    """Debugging function to apply process to input from webcam

    This function can be used to test a processing step on
    the input from the video capture device.  The output of the
    process is then displayed.  If nothing is passed, default 
    process is to just display the capture.

    Parameters
    ----------
    out : str
        The output file or None

    process : func or None
        If none, just display the input from the webcam.
        If provided, apply the function on each input frame and
        display post process result
    
    params : None or Dict
        Dictionary will be passed to the process method as a
        keyword arg.

    title : str
        The title for the display window
    """
    # ---------------------  create video capture  object  ------------------
    # video capture object
    vc = cv2.VideoCapture(0)
    if not vc.isOpened():
        raise IOError("Unable to open video capture")

    # ----------------------  prepare output file writer  -------------------
    if out is not None:
        assert isinstance(out, str),\
            "Expecting the string path of output file"

        # get dimensions of input from the webcam
        ret, frame = vc.read()

        # prepare output file
        out_file = cv2.VideoWriter(
            out,
            cv2.VideoWriter_fourcc('M', "J", "P", "G"),
            10,
            (frame.shape[1], frame.shape[0])
        )

    # tell user how to exit
    print("Hit the 'esc' key to exit")

    while True:
        # get frame from input video capture device
        ret, frame = vc.read()

        # -----------------------  process the image  -----------------------
        if process is None:
            # default to just showing the frames
            processed_frame = frame
         
        else:
            if params is None:
                # no parameters provided
                processed_frame = process(frame)

            else:
                assert isinstance(params, dict), "Expecting dict for params"
                # process params as keyword dict
                processed_frame = process(frame, **params)
        
        # -----------------------  display frame  ---------------------------
        cv2.imshow(title, processed_frame)
        if out is not None:
            out_file.write(processed_frame)
        # -------------------  exit on 'esc' key  ---------------------------
        if cv2.waitKey(1) == 27:
            break
    
    vc.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_webcam()