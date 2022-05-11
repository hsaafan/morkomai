from display import Display
from collections import deque
from itertools import cycle
import time
import os
from PIL import Image
import numpy as np


class ScreenRecorder:
    def __init__(self, display: Display,
                 capture_folder: str,
                 images_to_keep: int = 3) -> None:
        """Class used to record the screen.

        Parameters
        ----------
        display: Display
            The display to record.

        Attributes
        ----------
        display: Display
            The display that is being recorded.
        capture_folder: str
            The folder to store captures in.
        images_to_keep: int
            The number of images to keep available.
        """
        self.display = display
        self._image_data = deque(maxlen=images_to_keep)
        self._running = False
        curr_time = int(time.time())
        self._image_paths = [f'{capture_folder}/{curr_time}_{x}.png'
                             for x in range(images_to_keep)]

    def continuous_capture(self, delay: float = 0.1) -> None:
        """Take screen captures continuously.

        Parameters
        ----------
        delay: float, optional
            The number of seconds to wait in between captures.
        """
        self._running = True
        for path in cycle(self._image_paths):
            if not self._running:
                break
            self.display.capture(path)
            self._image_data.append(open_image(path))
            time.sleep(delay)
        for path in self._image_paths:
            if os.path.isfile(path):
                os.remove(path)

    def stop(self):
        """Signals the class to stop taking captures"""
        self._running = False

    def get_current_image(self) -> np.ndarray:
        """Returns the last captured image"""
        try:
            image = self._image_data[0]
        except TypeError:
            image = None
        return(image)


def open_image(path: str) -> np.ndarray:
    """Opens an image as a numpy array of values in [0, 1].

    Parameters
    ----------
    path: str
        The path of the image to open.
    """
    image = Image.open(path).convert('L')
    width, height = image.size
    image_data = np.asarray(image.getdata()).reshape((width, height))
    image_data = image_data.astype('float16') / 256
    return(image_data)


def images_similar(image1, image2, threshold: float = 0.05):
    """Check if two images are similar

    Parameters
    ----------
    image1
        The first image
    image2
        The second image
    threshold: float
        The threshold for the similarity metric.
    """
    if image1.size != image2.size:
        raise NotImplemented
    diff = image1 - image2
    rmse = (np.sum(diff ** 2) / image1.size) ** 0.5
    return(rmse < threshold)
