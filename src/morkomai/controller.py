import time
from collections import deque
import random

from PIL import ImageDraw, Image

from .globals import *
from .ai import AI
from .mortalkombat import MortalKombat
from .recorder import open_image_floats, open_image_array, convert_to_array
from . import vision
from .ml import MorkomAI


def start_game(AI_1: bool = False, AI_2: bool = False,
               char_1: str = None, char_2: str = None,
               AI_1_tactics: str = 'lstm',
               AI_2_tactics: str = 'lstm',
               randomize_characters: bool = False) -> None:
    """Function to quickly setup and start a game.

    Parameters
    ----------
    AI_1: bool, optional
        Add an AI to the game, defaults to False.
    AI_2: bool, optional
        Add a second AI to the game, defaults to False.
    char_1: str, optional
        The character AI_1 plays, if not set, will pick a random character. See
        CHARACTERS.
    char_2: str, optional
        The character AI_2 plays, if not set, will pick a random character. See
        CHARACTERS.
    AI_1_tactics: str, optional
        The tactics that AI 1 uses, defaults to 'random'.
    AI_2_tactics: str, optional
        The tactics that AI 1 uses, defaults to 'random'.
    randomize_characters: bool, optional
        Choose a random character after every match. Defaults to False.
    """
    game_controller = Controller(randomize_characters=randomize_characters)
    if AI_1:
        if char_1 is None:
            char_1 = CHARACTERS[random.randint(0, 6)]
        game_controller.add_ai(char_1, AI_1_tactics)
    if AI_2:
        if char_2 is None:
            char_2 = CHARACTERS[random.randint(0, 6)]
        game_controller.add_ai(char_2, AI_2_tactics)
    game_controller.start()


class Controller:
    """Game controller that manages the program.

    Attributes
    ----------
    game: MortalKombat
        The game the controller is attached to.
    key_queue: deque
        A queue of keys to press.
    toggle_key_queue: deque
        A queue of keys to toggle.
    pressed_keys: dict
        A list of controls that are currently toggled down.
    loop_delay: float
        The minimum time between loop iterations.
    key_delay: float
        The time to wait after pressing keys.
    vision: SpriteMatcher
        An object used to detect character sprites on screen.
    in_fight: bool
        Flag to mark whether the game is in a fight.
    last_scene: int
        The previous detected scene. See SCENES.
    randomize_characters: bool
        Choose a random character after every match if True.
    ai_players: list
        A list containing all AI players.
    use_vision: bool
        Is the controller using its vision.
    find_players: bool
        Is the controller looking for player sprites.
    vbounds_p1: tuple
        The previous bounding box of the player 1 sprite (x1, y1, x2, y2).
    vbounds_p2: tuple
        The previous bounding box of the player 2 sprite (x1, y1, x2, y2).
    n_rounds: int
        A count of how many rounds the controller has played.
    lstm_model: MorkomAI
        The LSTM model to use for choosing actions.

    Parameters
    ----------
    randomize_characters: bool, optional
        Choose a random character after every match. Defaults to False.
    """
    def __init__(self, randomize_characters: bool = False) -> None:
        # Game
        self.game = MortalKombat()
        self.key_queue = deque()
        self.toggle_key_queue = deque()
        self.pressed_keys = dict()
        self.loop_delay = 0.1
        self.key_delay = 0.05
        # AI Vision
        vision.load_spritedb(self.game.sprite_folder)
        fight_prompt = open_image_array(self.game.fight_prompt)
        self._scene_templates = {
            'join_screen': open_image_floats(self.game.join_screen),
            'char_select': open_image_floats(self.game.character_screen),
            'continue_screen': open_image_floats(self.game.continue_screen),
            'fight_prompt': fight_prompt[:, :, 0] / 255,
        }
        self._timer_image = open_image_array(self.game.timer_image)[:, :, 0]
        self.vision = vision.SpriteMatcher(0, 0, timeout=0.25)
        self.reset_vision_bounds()
        self._vision_override = False
        self._fight_vision_override = False
        self.in_fight = False
        self.last_scene = SCENES['OTHER']
        # AI
        self.randomize_characters = randomize_characters
        self.ai_players = []
        self.lstm_model = MorkomAI(self.game.model_folder)
        self.n_rounds = self.lstm_model.n_rounds

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

    def queue_toggle_keystroke(self, keystroke: tuple) -> None:
        """Add a keystroke to the key toggle queue.

        Parameters
        ----------
        keystroke: tuple
            0 (str): The keystroke to toggle.
            1 (bool): True to press down, False to press up.
        """
        self.toggle_key_queue.append(keystroke)
        if keystroke[1]:
            self.pressed_keys[keystroke[0]] = True
        else:
            try:
                del self.pressed_keys[keystroke[0]]
            except KeyError:
                pass

    def clear_key_queues(self) -> None:
        """Clears the key queues and raises any keys currently down."""
        self.key_queue = deque()
        self.toggle_key_queue = deque()
        for key in list(self.pressed_keys.keys()):
            self.queue_toggle_keystroke((key, False))
        self._press_keys()

    def add_ai(self, char: str, tactics: str = 'random') -> None:
        """Add an AI to the controller.

        Parameters
        ----------
        char: str
            The character to choose. See CHARACTERS.
        tactics: str, optional
            The tactics the AI should use. Defaults to 'random'.
            Available tactics:
                - 'random'
                - 'lstm'
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
            Available tactics:
                - 'random'
                - 'lstm'
        """
        if tactics == 'random':
            fight_fun = self.ai_players[player].random_moves
        elif tactics == 'lstm':
            fight_fun = self.ai_players[player].lstm_ai
        else:
            raise ValueError(f'No tactics found called "{tactics}"')
        self.ai_players[player].fight = fight_fun

    def set_chars(self, p1_char: str, p2_char: str = None) -> None:
        """Set the characters for the AI players.

        Parameters
        ----------
        p1_char: str
            The character that the first AI should choose. See CHARACTERS.
        p2_char: str, optional
            The character that the second AI should choose. See CHARACTERS.
        """
        if len(self.ai_players) > 0:
            self.ai_players[0].character = p1_char
        if len(self.ai_players) > 1:
            self.ai_players[1].character = p2_char
        self.vision.change_chars(CHARACTER_IDS[p1_char],
                                 CHARACTER_IDS[p2_char])

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
        x1m, y1m, x2m, y2m = PLAY_AREA
        self.vbounds_p1 = [max(self.vbounds_p1[0] - xp, x1m),
                           max(self.vbounds_p1[1] - yp, y1m),
                           min(self.vbounds_p1[2] + xp2, x2m),
                           min(self.vbounds_p1[3] + yp2, y2m)]
        self.vbounds_p2 = [max(self.vbounds_p2[0] - xp, x1m),
                           max(self.vbounds_p2[1] - yp, y1m),
                           min(self.vbounds_p2[2] + xp2, x2m),
                           min(self.vbounds_p2[3] + yp2, y2m)]

    def reset_vision_bounds(self) -> None:
        """Resets bounds to starting positons."""
        self.vbounds_p1 = START_POSITIONS[0]
        self.vbounds_p2 = START_POSITIONS[1]

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
        self.clear_key_queues()
        for player in self.ai_players:
            player.state = state

    def set_AI_enemy_chars(self, scene_img: Image.Image) -> None:
        """Sets the enemy_character variables for AI players.

        Uses the names on the health bars to detect players. Will also update
        the AI player character attributes if there is a mismatch.

        Parameters
        ----------
        scene_img: PIL.Image.Image
            An image of the scene.
        """
        scene = vision.convert_to_array(scene_img)
        plates = self.game.nameplate_folder
        p1_char = vision.detect_character(scene, 0, plates)
        p2_char = vision.detect_character(scene, 1, plates)
        if len(self.ai_players) == 2:
            p1_set_char = self.ai_players[0].character
            p2_set_char = self.ai_players[1].character
            if p1_char != p1_set_char:
                self.ai_players[0].character = p1_char
            if p2_char != p2_set_char:
                self.ai_players[1].character = p2_char
            self.ai_players[0].enemy_character = p2_char
            self.ai_players[1].enemy_character = p1_char
        elif len(self.ai_players) == 1:
            p1_set_char = self.ai_players[0].character
            if p1_char != p1_set_char:
                self.ai_players[0].character = p1_char
            self.ai_players[0].enemy_character = p2_char
        print(f'Fight started: {p1_char} vs {p2_char}')
        self.set_chars(p1_char, p2_char)

    def reset_AI_lstm_states(self) -> None:
        """Resets the lstm states of each AI."""
        for player in self.ai_players:
            player.reset_lstm_state()

    def end_of_match(self) -> None:
        """Run at the end of every match."""
        self.n_rounds += 1
        self.lstm_model.n_rounds = self.n_rounds
        print(f'End of fight. Total rounds: {self.n_rounds}')
        self.clear_key_queues()
        self.set_AI_states(AI_STATES['INTERMISSION'])
        self.reset_vision_bounds()
        for player in self.ai_players:
            player.update_lstm_memory()
        self.in_fight = False

    def detect_scene(self, scene_img: Image.Image) -> None:
        """Detect the current scene on screen.

        Parameters
        ----------
        scene_img: PIL.Image.Image
            An image of the scene.
        """
        scene = vision.detect_current_scene(scene_img, **self._scene_templates)
        if scene == SCENES['INTRODUCTION'] != self.last_scene:
            self.set_AI_states(AI_STATES['JOINING'])
        elif scene == SCENES['CHARACTER_SELECT'] != self.last_scene:
            if self.randomize_characters:
                self.set_chars(CHARACTERS[random.randint(0, 6)],
                               CHARACTERS[random.randint(0, 6)])
            self.set_AI_states(AI_STATES['SELECT_CHARACTER'])
        elif scene == SCENES['FIGHT_PROMPT'] != self.last_scene:
            self.in_fight = True
            self.vision.update_image(scene_img)
            self.vision.update_environment_palette()
            self.set_AI_enemy_chars(scene_img)
            self.reset_AI_lstm_states()
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
        self.vision.update_image(img)
        if self.find_players:
            char1, char2 = [self.vision._p1_char, self.vision._p2_char]
            p1 = self.vision.find_players(self.vbounds_p1, only_p1=True)[0]
            p2 = self.vision.find_players(self.vbounds_p2, only_p2=True)[0]
            if p1 is not None:
                p1_sprite, self.vbounds_p1 = p1
                p1_sprite_desc = vision.SpriteDB.sprites[char1][p1_sprite].desc
                # Remove number and reversed sign at end of description
                if p1_sprite_desc[-3:] == '(R)':
                    p1_sprite = SPRITE_IDS[p1_sprite_desc[:-4]]
                else:
                    p1_sprite = SPRITE_IDS[p1_sprite_desc[:-1]]
            else:
                p1_sprite = SPRITE_IDS['idle']
            if p2 is not None:
                p2_sprite, self.vbounds_p2 = p2
                p2_sprite_desc = vision.SpriteDB.sprites[char2][p2_sprite].desc
                if p2_sprite_desc[-3:] == '(R)':
                    p2_sprite = SPRITE_IDS[p2_sprite_desc[:-4]]
                else:
                    p2_sprite = SPRITE_IDS[p2_sprite_desc[:-1]]
            else:
                p2_sprite = SPRITE_IDS['idle']
            info[0] += [p1_sprite, self.vbounds_p1]
            info[1] += [p2_sprite, self.vbounds_p2]
            self.expand_vision_bounds()
            img_array = self.vision.img
        else:
            img_array = convert_to_array(img)
        p1_health, p2_health = vision.get_health(img_array)
        timed_out = vision.is_timer_done(img_array, self._timer_image)
        if p1_health <= 0 or p2_health <= 0 or timed_out:
            self.end_of_match()
        info[0][0] = p1_health
        info[1][0] = p2_health
        return(info)

    def save_tagged_img(self, img: Image.Image, info: list) -> None:
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
        draw = ImageDraw.Draw(img)
        if p1_info[1] > -1:
            text1 = f'{SPRITE_DESC[p1_info[1]]}: {p1_info[0]}'
            bounds1 = p1_info[2]
            draw.text((bounds1[0], bounds1[1] - 5), text1)
        draw.rectangle(list(self.vbounds_p1), outline="blue", width=3)
        if p2_info[1] > -1:
            text2 = f'{SPRITE_DESC[p2_info[1]]}: {p2_info[0]}'
            bounds2 = p2_info[2]
            draw.text((bounds2[0], bounds2[1] - 5), text2)
        draw.rectangle(list(self.vbounds_p2), outline="red", width=3)
        img.save(f'.captures/{int(time.time())}.png')

    def _press_keys(self) -> None:
        """Presses all keys in the key queue"""
        while self.key_queue:
            key = self.key_queue.popleft()
            self.game._dosbox.keystroke(key)
            time.sleep(self.key_delay)
        while self.toggle_key_queue:
            toggle_key = self.toggle_key_queue.popleft()
            self.game._dosbox.toggle_key(toggle_key)
            time.sleep(self.key_delay)

    def _main_loop(self) -> None:
        """The main controller loop."""
        player_sequence = list(range(len(self.ai_players)))
        while True:
            loop_start = time.time()
            self._press_keys()
            self.img = self.game._recorder.step()
            if self.use_vision:
                if self.in_fight:
                    info = self.grab_fight_info(self.img)
                    if self.game.save_captures:
                        self.save_tagged_img(self.img, info)
                    for player in self.ai_players:
                        player.update_info(info)
                else:
                    self.detect_scene(self.img)
                    for player in self.ai_players:
                        player.reset_info()

            random.shuffle(player_sequence)
            for i in player_sequence:
                self.ai_players[i].step()
            delay = self.loop_delay - (time.time() - loop_start)
            if delay > 0:
                time.sleep(delay)
