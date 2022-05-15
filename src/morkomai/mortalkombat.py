from os.path import abspath
import threading
import yaml
from .dosbox import DOSBox
from .recorder import ScreenRecorder


class MortalKombat:
    def __init__(self, dosbox: DOSBox = None, recorder: ScreenRecorder = None,
                 settings: dict = None) -> None:
        """Class used to run Mortal Kombat application.

        Parameters
        ----------
        dosbox: DOSBox, optional
            The DOSBox object to run Mortal Kombat on. If none are passed, a
            new DOSBox object (along with a new Display object) will be
            created.
        recorder: ScreenRecorder, optional
            An object that is used to take snapshots of the screen. If none are
            passed, a ScreenRecorder object will be created and attached to the
            dosbox display.
        settings: dict, optional
            A dictionary containing all required settings, if none are passed,
            the settings.yaml file will be loaded.

        Attributes
        ----------
        capture_folder: str
            The folder to store image captures in.
        join_screen: str
            The path of an image of the screen where players can join.
        character_select_screen: str
            The path of an image of the character select screen.
        fight_prompt: str
            Path to an image of the fight prompt.
        p_save: float
            The probability of saving captured images to file.
        record_framerate: float
            The number of frames to capture per second.
        """
        if dosbox is None:
            dosbox = DOSBox()
        self._dosbox = dosbox
        self.load_settings(settings)

        if recorder is None:
            recorder = ScreenRecorder(dosbox._display, self.capture_folder,
                                      self.p_save, self.record_framerate)
        self._recorder = recorder

    def start(self, conf_file: str = 'dos.conf') -> None:
        """Starts Mortal Kombat.

        Parameters
        ----------
        conf_file: str, optional
            The path of the dosbox conf file that should include startup
            commands for game. Default conf file included will be run if no
            other are passed.
        """
        if self._dosbox.is_running:
            raise RuntimeError('dosbox is already running')
        if self._recorder.is_running:
            raise RuntimeError('Recorder is already running')
        self._dosbox.start(conf_file=conf_file)
        threading.Thread(target=self._recorder.start).start()

    def load_settings(self, settings: dict = None):
        """Loads a dict containing program settings.

        Parameters
        ----------
        settings: dict
            If None, will load settings.yaml file. Takes the following keys:
                capture_folder: str
                    The folder to store game screenshots in
                join_screen: str
                    Path to an image that signifies that players can start
                    joining.
                character_screen: str
                    Path to an image that signifies that players can start
                    to select characters.
                fight_prompt: str
                    Path to an image of the fight prompt.
                p_save: float
                    Probability of saving screenshots to file.
                record_framerate: float
                    The number of frames to capture per second.
        """
        if settings is None:
            with open('settings.yaml') as f:
                settings = yaml.safe_load(f)
        self.capture_folder = abspath(settings['capture_folder'])
        self.join_screen = abspath(settings['join_screen'])
        self.character_screen = abspath(settings['character_screen'])
        self.fight_prompt = abspath(settings['fight_prompt'])
        self.p_save = settings['p(save_image)']
        self.record_framerate = settings['record_framerate']

    def join(self, player: int) -> None:
        """Start players that are controlled by this program."""
        if player == 0:
            self._dosbox.keystroke('F1')
        if player == 1:
            self._dosbox.keystroke('F2')

    def stop(self) -> None:
        """Stop Mortal Kombat and the dosbox application."""
        self._recorder.stop()
        self._dosbox.stop()
