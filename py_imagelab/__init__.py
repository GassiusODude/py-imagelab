import logging
import colorlog

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def enable_color_logging(level :int = logging.INFO) -> None:
    """
    Enable color logging
    """
    # prevent duplicate handlers if the user calls this twice
    if any(isinstance(h, colorlog.StreamHandler) for h in logger.handlers):
        return

    # configure the handler
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    ))

    # attach to the library logger
    logger.addHandler(handler)
    logger.setLevel(level)

    # (Optional) Stop propagation if you want your library logs
    # to ONLY use your color formatter and NOT the user's root logger.
    logger.propagate = False

