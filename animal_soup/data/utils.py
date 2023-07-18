from pathlib import Path
from torchvision.io import VideoReader


def get_video_metadata(vid_path: Path):
    """ Simple wrapper to get video availability, width, height, and frame number """
    try:
        with VideoReader(src=vid_path) as reader:
            framenum = reader.nframes
            frame = next(reader)
            width = frame.shape[1]
            height = frame.shape[0]
            ret = True
    except BaseException as e:
        ret = False
        print(e)
        print('Error reading file {}'.format(videofile))
    return ret, width, height, framenum