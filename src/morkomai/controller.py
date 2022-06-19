import time
from collections import deque
import random

import numpy as np
from PIL import ImageDraw, Image

from .mortalkombat import MortalKombat
from .enumerations import AI_STATES, SCENES, START_POSITIONS
from .ai import AI
from . import vision
from .recorder import open_image, open_image_floats


def start_game(AI_1: bool = False, AI_2: bool = False,
               AI_1_tactics: str = 'random',
               AI_2_tactics: str = 'random') -> None:
    """Function to quickly setup and start a game.

    Parameters
    ----------
    AI_1: bool, optional
        Add an AI to the game, defaults to False.
    AI_2: bool, optional
        Add a second AI to the game, defaults to False.
    AI_1_tactics: str, optional
        The tactics that AI 1 uses, defaults to 'random'.
    AI_2_tactics: str, optional
        The tactics that AI 1 uses, defaults to 'random'.
    """
    game_controller = Controller()
    if AI_1:
        game_controller.add_ai(0, AI_1_tactics)
    if AI_2:
        game_controller.add_ai(0, AI_2_tactics)
    game_controller.start()


class Controller:
    def __init__(self) -> None:
        """Game controller that manages the program.

        Attributes
        ----------
        game: MortalKombat
            The game the controller is attached to.
        key_queue: deque
            A queue of keys to press.
        loop_delay: float
            The minimum time between loop iterations.
        key_delay: float
            The time to wait after pressing keys.
        vision: SpriteMatcher
            An object used to detect character sprites on screen.
        vbounds_p1: tuple
            The previous bounding box of the player 1 sprite (x1, y1, x2, y2).
        vbounds_p2: tuple
            The previous bounding box of the player 2 sprite (x1, y1, x2, y2).
        use_vision: bool
            Is the controller using its vision.
        find_players: bool
            Is the controller looking for player sprites.
        in_fight: bool
            Flag to mark whether the game is in a fight.
        last_scene: int
            The previous detected scene. See SCENES.
        ai_players: list
            A list containing all ai players.
        """
        # Game
        self.game = MortalKombat()
        self.key_queue = deque()
        self.loop_delay = 0.1
        self.key_delay = 0.2
        # AI Vision
        vision.load_all_sprites()
        fight_prompt = open_image(self.game.fight_prompt)
        self._scene_templates = {
            'join_screen': open_image_floats(self.game.join_screen),
            'char_select': open_image_floats(self.game.character_screen),
            'fight_prompt': np.asarray(fight_prompt.getdata())[:, 0] / 255
        }
        self.vision = vision.SpriteMatcher(0, 0, timeout=0.25)
        self.vbounds_p1 = START_POSITIONS[0]
        self.vbounds_p2 = START_POSITIONS[1]
        self._vision_override = False
        self._fight_vision_override = False
        self.in_fight = False
        self.last_scene = SCENES['OTHER']
        # AI
        self.ai_players = []

    def _get_use_vision(self) -> bool:
        if self._vision_override:
            return(True)
        if len(self.ai_players) > 0:
            return(True)
        return(False)

    def _set_use_vision(self, value: bool) -> None:
        self._vision_override = value
    use_vision = property(fget=_get_use_vision, fset=_set_use_vision,
                          doc="""Is the controller using AI vision.""")

    def _get_find_players(self) -> bool:
        if self._fight_vision_override:
            return(True)
        if len(self.ai_players) > 0:
            if self.ai_players[0].state == AI_STATES['FIGHTING']:
                return(True)
        return(False)

    def _set_find_players(self, value: bool) -> None:
        self._fight_vision_override = value
    find_players = property(fget=_get_find_players, fset=_set_find_players,
                            doc="""Is the controller matching sprites.""")

    def queue_keystroke(self, keystroke: str) -> None:
        """Add a keystroke to the key queue.

        Parameters
        ----------
        keystroke: str
            The keystroke to add.
        """
        self.key_queue.append(keystroke)

    def add_ai(self, char: int, tactics: str = 'random') -> None:
        """Add an AI to the controller.

        Parameters
        ----------
        char: int
            The character to choose.
        tactics: str, optional
            The tactics the AI should use.
        """
        player = len(self.ai_players)
        self.ai_players.append(AI(self, player, char, False))
        self.set_tactics(player, tactics)

    def set_tactics(self, player: int, tactics: str) -> None:
        """Set the tactics of an AI.

        Parameters
        ----------
        player: int
            The AI to set the tactics for.
        tacitcs: str
            The tactics the AI should use.
        """
        if tactics == 'random':
            fight_fun = self.ai_players[player].random_moves
        else:
            raise ValueError(f'No tactics found called "{tactics}"')
        self.ai_players[player].fight = fight_fun

    def set_chars(self, p1_char: int, p2_char: int = None) -> None:
        """Set the characters for the AI players.

        Parameters
        ----------
        p1_char: int
            The character that the first AI should choose.
        p2_char: int, optional
            The character that the second AI should choose.
        """
        if len(self.ai_players) > 0:
            self.ai_players[0].character = p1_char
        if len(self.ai_players) > 1:
            self.ai_players[1].character = p2_char
        self.vision.change_chars(p1_char, p2_char)

    def expand_vision_bounds(self, xp: int = 20, yp: int = 10,
                             xp2: int = 20, yp2: int = 10) -> None:
        """Expand the player bounding boxes for the sprite search.

        Parameters
        ----------
        xp: int, optional
            The pixels the bounding boxes should be expanded left.
        yp: int, optional
            The pixels the bounding boxes should be expanded up.
        xp2: int, optional
            The pixels the bounding boxes should be expanded right.
        yp2: int, optional
            The pixels the bounding boxes should be expanded down.
       """
        self.vbounds_p1 = [max(self.vbounds_p1[0] - xp, 160),
                           max(self.vbounds_p1[1] - yp, 140),
                           min(self.vbounds_p1[2] + xp2, 480),
                           min(self.vbounds_p1[3] + yp2, 340)]
        self.vbounds_p2 = [max(self.vbounds_p2[0] - xp, 160),
                           max(self.vbounds_p2[1] - yp, 140),
                           min(self.vbounds_p2[2] + xp2, 480),
                           min(self.vbounds_p2[3] + yp2, 340)]

    def start(self) -> None:
        """Start the controller."""
        self.game.start()
        self._main_loop()

    def set_AI_states(self, state: int) -> None:
        """Set the states of all the AI players.

        Parameters
        ----------
        state: int
            The state to set. See AI_STATES.
        """
        for player in self.ai_players:
            player.state = state

    def detect_scene(self, scene: Image.Image) -> None:
        """Detect the current scene on screen.

        Parameters
        ----------
        scene: PIL.Image.Image
            An image of the scene.
        """
        scene = vision.detect_current_scene(scene, **self._scene_templates)
        if scene == SCENES['INTRODUCTION'] != self.last_scene:
            self.set_AI_states(AI_STATES['JOINING'])
        elif scene == SCENES['CHARACTER_SELECT'] != self.last_scene:
            self.set_AI_states(AI_STATES['SELECT_CHARACTER'])
        elif scene == SCENES['FIGHT_PROMPT'] != self.last_scene:
            self.in_fight = True
            self.set_AI_states(AI_STATES['FIGHTING'])
        self.last_scene = scene

    def grab_fight_info(self, img: Image.Image) -> list:
        """Extract the fight information from the current screen.

        Parameters
        ----------
        scene: PIL.Image.Image
            An image of the scene.

        Returns
        -------
        info: list
            List with player 1 info at index 0 and player 2 info at index 2.
            The player information is a list ordered as follows.
                Health: float
                    Player health from 0-100.
                Sprite ID: int
                    The current sprite of the player.
                Bounding Box: tuple
                    The bounding box of the player sprite (x1, y1, x2, y2).
        """
        info = [[0], [0]]
        if self.find_players:
            self.vision.update_image(img)
            p1 = self.vision.find_players(self.vbounds_p1, only_p1=True)[0]
            p2 = self.vision.find_players(self.vbounds_p2, only_p2=True)[0]
            if p1 is not None:
                p1_sprite, self.vbounds_p1 = p1
            else:
                p1_sprite = -1
            if p2 is not None:
                p2_sprite, self.vbounds_p2 = p2
            else:
                p2_sprite = -1
            info[0] += [p1_sprite, self.vbounds_p1]
            info[1] += [p2_sprite, self.vbounds_p2]
            self.expand_vision_bounds()
            img_array = self.vision.img
        else:
            width, height = img.size
            c = len(img.mode)
            img_array = np.asarray(img.getdata()).reshape((width, height, c))
        p1_health, p2_health = vision.get_health_bars(img_array)
        if p1_health <= 0 or p2_health <= 0:
            self.set_AI_states(AI_STATES['INTERMISSION'])
            self.in_fight = False
        info[0][0] = p1_health
        info[1][0] = p2_health
        return(info)

    def save_tagged_img(self, img: Image.Image, info: str) -> None:
        """Save an image to disk with overlayed bounding boxes.
        
        Parameters
        ----------
        img: PIL.Image.Image
            The image to overlay the bounding boxes on.
        info: list
            List with player 1 info at index 0 and player 2 info at index 2.
            The player information is a list ordered as follows.
                Health: float
                    Player health from 0-100.
                Sprite ID: int
                    The current sprite of the player.
                Bounding Box: tuple
                    The bounding box of the player sprite (x1, y1, x2, y2).
        """
        p1_info, p2_info = info
        char1, char2 = [self.vision._p1_char, self.vision._p2_char]
        draw = ImageDraw.Draw(img)
        if p1_info[1] > -1:
            text1 = vision.SpriteDB.sprites[char1][p1_info[1]].desc
            bounds1 = p1_info[2]
            draw.text((bounds1[0], bounds1[1] - 5), text1)
        draw.rectangle(list(self.vbounds_p1), outline="blue", width=3)
        if p2_info[1] > -1:
            text2 = vision.SpriteDB.sprites[char2][p2_info[1]].desc
            bounds2 = p2_info[2]
            draw.text((bounds2[0], bounds2[1] - 5), text2)
        draw.rectangle(list(self.vbounds_p2), outline="red", width=3)
        img.save(f'.captures/{int(time.time())}.png')

    def _main_loop(self) -> None:
        """The main controller loop."""
        player_sequence = list(range(len(self.ai_players)))
        while True:
            for i in range(len(self.key_queue)):
                self.game._dosbox.keystroke(self.key_queue.popleft())
                time.sleep(self.key_delay)
            loop_start = time.time()
            img = self.game._recorder.step()
            if self.use_vision:
                if self.in_fight:
                    info = self.grab_fight_info(img)
                    self.save_tagged_img(img, info)
                    for player in self.ai_players:
                        player.update_info(info)
                else:
                    self.detect_scene(img)

            random.shuffle(player_sequence)
            for i in player_sequence:
                self.ai_players[i].step()
            delay = self.loop_delay - (time.time() - loop_start)
            if delay > 0:
                time.sleep(delay)
