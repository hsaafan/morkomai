from os.path import abspath
import yaml
from .dosbox import DOSBox
from .recorder import ScreenRecorder


class MortalKombat:
    """Class used to run Mortal Kombat application.

    Attributes
    ----------
    capture_folder: str
        The folder to store image captures in.
    save_captures: bool
        Save the captures with bounding boxes to disk.
    join_screen: str
        The path of an image of the screen where players can join.
    character_select_screen: str
        The path of an image of the character select screen.
    continue_screen: str
        The path of an image of the continue screen.
    fight_prompt: str
        Path to an image of the fight prompt.
    timer_image: str
        Path to an image of the timer at 00.
    p_save: float
        The probability of saving captured images to file.
    sprite_folder: str
        The folder that the sprites are stored in.
    nameplate_folder: str
        The folder that the name plates are stored in.
    model_folder: str
        The folder to store ML models in.

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
    """
    def __init__(self, dosbox: DOSBox = None, recorder: ScreenRecorder = None,
                 settings: dict = None) -> None:
        if dosbox is None:
            dosbox = DOSBox()
        self._dosbox = dosbox
        self.load_settings(settings)

        if recorder is None:
            recorder = ScreenRecorder(dosbox._display, self.capture_folder,
                                      self.p_save)
        self._recorder = recorder

    def start(self, conf_file: str = 'dos.conf') -> None:
        """Starts DOSBox and Mortal Kombat.

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
        self._recorder.start()

    def load_settings(self, settings: dict = None):
        """Loads a dict containing program settings.

        Parameters
        ----------
        settings: dict
            If None, will load settings.yaml file. Takes the following keys:
                capture_folder: str
                    The folder to store game screenshots in.
                save_captures: bool
                    Save the captures with bounding boxes to disk.
                join_screen: str
                    Path to an image that signifies that players can start
                    joining.
                character_screen: str
                    Path to an image that signifies that players can start
                    to select characters.
                continue_screen: str
                    The path of an image of the continue screen.
                fight_prompt: str
                    Path to an image of the fight prompt.
                timer_image: str
                    Path to an image of the timer at 00.
                p_save: float
                    Probability of saving screenshots to file.
                sprite_folder: str
                    The folder that the sprites are stored in.
                nameplate_folder: str
                    The folder that the name plates are stored in.
                model_folder: str
                    The folder to store ML models in.
        """
        if settings is None:
            with open('settings.yaml') as f:
                settings = yaml.safe_load(f)
        self.capture_folder = abspath(settings['capture_folder'])
        self.save_captures = settings['save_captures']
        self.join_screen = abspath(settings['join_screen'])
        self.character_screen = abspath(settings['character_screen'])
        self.continue_screen = abspath(settings['continue_screen'])
        self.fight_prompt = abspath(settings['fight_prompt'])
        self.timer_image = abspath(settings['timer_image'])
        self.p_save = settings['p(save_image)']
        self.sprite_folder = abspath(settings['sprite_folder'])
        self.nameplate_folder = abspath(settings['nameplate_folder'])
        self.model_folder = abspath(settings['model_folder'])

    def stop(self) -> None:
        """Stop Mortal Kombat and the dosbox application."""
        self._recorder.stop()
        self._dosbox.stop()
