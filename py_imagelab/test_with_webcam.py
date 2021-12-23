import cv2
import os


def test_webcam(out=None, process=None, params=None, title="Preview", cap=0):
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
    
    cap : int or str
        The capture device.  Default to 0 for webcam.
        If string, could be a video file
    """
    # ---------------------  create video capture  object  ------------------
    # video capture object
    if isinstance(cap, str):
        assert os.path.isfile(cap), "Cap is not a valid file"

    vc = cv2.VideoCapture(cap)
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
        detections = None
        if process is None:
            # default to just showing the frames
            processed_frame = frame
         
        else:
            if params is None:
                # no parameters provided
                processed_frame, detections = process(frame)

            else:
                assert isinstance(params, dict), "Expecting dict for params"
                # process params as keyword dict
                processed_frame, detections = process(frame, **params)
        
        # -----------------------  display frame  ---------------------------
        # update detection boxes into the image
        if detections is not None:
            for (x,y,w,h) in detections:
                processed_frame = cv2.rectangle(processed_frame,
                    (x,y), (x+w, y+h), color=(200,0,0), thickness=10)

        # display image
        cv2.imshow(title, processed_frame)

        if out is not None:
            out_file.write(processed_frame)
        # -------------------  exit on 'esc' key  ---------------------------
        if cv2.waitKey(1) == 27:
            break
    
    vc.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("out", default="", help="Output file")
    parser.add_argument(
        "--cap", default="", 
        help="Video file.  If not provided, use webcam as input")
    args = parser.parse_args()

    assert len(args.out) > 0, "Expecting an output file"
    if args.cap == "":
        cap = 0
    else:
        cap = args.cap

    test_webcam(out=args.out, cap=cap)