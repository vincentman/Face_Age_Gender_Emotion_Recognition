import cv2
from collections import namedtuple

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


VideoProps = namedtuple('VideoProps', ['width', 'height', 'fourcc', 'fps'])
def get_input_video_file_props(capture):
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = capture.get(cv2.CAP_PROP_FOURCC)
    codec = decode_fourcc(fourcc)
    fps = capture.get(cv2.CAP_PROP_FPS)
    return VideoProps(width, height, fourcc, fps)
