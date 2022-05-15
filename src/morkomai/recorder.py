import time
import random
from collections import deque
from PIL import Image
import numpy as np
import mss
from mss.screenshot import ScreenShot
from .display import Display


class ScreenRecorder:
    def __init__(self, display: Display,
                 capture_folder: str,
                 p_save: float = 0,
                 record_framerate: float = 24,
                 images_to_keep: int = 3) -> None:
        """Class used to record the screen.

        Parameters
        ----------
        display: Display
            The display to record.
        capture_folder: str
            The folder to store captures in.
        p_save: float, optional
            The probability of saving captured images to file.
        delay: float, optional
            The number of screenshots to take per second. Defaults to 24.
        images_to_keep: int, optional
            The number of images to keep available. Defaults to 3.

        Attributes
        ----------
        display: Display
            The display that is being recorded.
        is_running: bool
            True if currently recording screen.
        capture_folder: str
            The folder to store captures in.
        p_save: float
            The probability of saving captured images to file.
        delay: float
            The delay in seconds between screen captures.
        """
        self.display = display
        self.capture_folder = capture_folder
        self.p_save = p_save
        self.delay = 1 / record_framerate
        self._image_data = deque(maxlen=images_to_keep)
        self.is_running = False
        self._mss = None

    def start(self) -> None:
        """Take screen captures continuously."""
        self.is_running = True
        # Create mss screenshot object and attach it to display
        if not self.display.is_running:
            raise RuntimeError('Display is not running, recorder cannot start')
        self._mss = mss.mss(display=f':{self.display.display_id}')
        time.sleep(0.01)
        self._mss_monitor = self._mss.monitors[1]
        while True:
            if not self.is_running:
                break
            self._image_data.append(self.capture())
            self.random_capture(self.p_save)
            time.sleep(self.delay)

    def stop(self) -> None:
        """Signals the class to stop taking captures."""
        self._running = False

    def get_current_image(self) -> Image:
        """Returns the last captured image."""
        try:
            sct = self._image_data[0]
            image = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        except TypeError:
            image = None
        return(image)

    def get_current_image_floats(self) -> np.ndarray:
        """Converts the last image into a float array and returns it."""
        image = self.get_current_image()
        if image is not None:
            return(convert_to_floats(image))

    def save_current_image(self, filename: str) -> None:
        """Save the last captured image in the captures folder.

        Parameters
        ----------
        filename: str
            The name of the image file (should include extension).
        """
        image = self.get_current_image()
        if image is not None:
            image.save(f'{self.capture_folder}/{filename}')

    def random_capture(self, p: float) -> None:
        """Roll a value in [0, 1], if its greater than p, save a capture.

        Saves the screenshot to the capture folder set in settings.yaml.

        Parameters
        ----------
        p: float
            The probability of taking a screenshot.
        """
        if random.random() > (1 - p):
            file_name = int(time.time())
            self.save_current_image(f'{file_name}.png')

    def capture(self) -> ScreenShot:
        """Capture the display and save it to a file."""
        return(self._mss.grab(self._mss_monitor))


def open_image(path: str) -> Image:
    """Opens an image as a PIL Image object.

    Parameters
    ----------
    path: str
        The path of the image to open.
    """
    image = Image.open(path)
    return(image)


def convert_to_floats(image: Image) -> np.ndarray:
    """Converts a PIL Image object into a numpy array of floats in [0, 1].

    Parameters
    ----------
    image: Image
        The PIL image to convert.
    """
    image = image.convert('L')
    width, height = image.size
    image_data = np.asarray(image.getdata()).reshape((width, height))
    image_data = image_data.astype('float16') / 256
    return(image_data)


def open_image_floats(path: str) -> np.ndarray:
    """Opens an image as a numpy array of floats in [0, 1].

    Parameters
    ----------
    path: str
        The path of the image to open.
    """
    image = open_image(path)
    image_data = convert_to_floats(image)
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
