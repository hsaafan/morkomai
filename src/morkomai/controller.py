import time
from collections import deque
import random

import numpy as np
from PIL import ImageDraw, Image

from .mortalkombat import MortalKombat
from .ai import AI, AI_STATES
from . import vision
from .recorder import open_image, open_image_floats


def start_game(AI_1: bool = False, AI_2: bool = True,
               AI_1_tactics: str = 'random',
               AI_2_tactics: str = 'random') -> MortalKombat:
    game_controller = Controller()
    if AI_1:
        game_controller.add_ai(0, 0.1, AI_1_tactics)
    if AI_2:
        game_controller.add_ai(0, 0.1, AI_2_tactics)
    game_controller.start()


class Controller:
    def __init__(self) -> None:
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
        self.vbounds_p1 = vision.START_POSITIONS[0]
        self.vbounds_p2 = vision.START_POSITIONS[1]
        self._vision_override = False
        self._fight_vision_override = False
        self.in_fight = False
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
        self.key_queue.append(keystroke)

    def add_ai(self, char: int, move_speed: float,
               tactics: str = 'random') -> int:
        player = len(self.ai_players)
        self.ai_players.append(AI(self, player, char, move_speed, False))
        self.set_tactics(player, tactics)
        return(player)

    def set_tactics(self, player: int, tactics: str) -> None:
        if tactics == 'random':
            fight_fun = self.ai_players[player].random_moves
        else:
            raise ValueError(f'No tactics found called "{tactics}"')
        self.ai_players[player].fight = fight_fun

    def set_chars(self, p1_char: int, p2_char: int) -> None:
        if len(self.ai_players) > 0:
            self.ai_players[0].character = p1_char
        if len(self.ai_players) > 1:
            self.ai_players[1].character = p2_char
        self.vision.change_chars(p1_char, p2_char)

    def expand_vision_bounds(self, xp: int = 20, yp: int = 10,
                             xp2: int = 20, yp2: int = 10) -> None:
        self.vbounds_p1 = [max(self.vbounds_p1[0] - xp, 160),
                           max(self.vbounds_p1[1] - yp, 140),
                           min(self.vbounds_p1[2] + xp2, 480),
                           min(self.vbounds_p1[3] + yp2, 340)]
        self.vbounds_p2 = [max(self.vbounds_p2[0] - xp, 160),
                           max(self.vbounds_p2[1] - yp, 140),
                           min(self.vbounds_p2[2] + xp2, 480),
                           min(self.vbounds_p2[3] + yp2, 340)]

    def start(self) -> None:
        self.game.start()
        self._main_loop()

    def set_AI_states(self, state: int) -> None:
        for player in self.ai_players:
            player.state = state

    def detect_scene(self, scene: Image.Image) -> None:
        scene = vision.detect_current_scene(scene, **self._scene_templates)
        if scene == vision.SCENES['INTRODUCTION']:
            self.set_AI_states(AI_STATES['JOINING'])
        elif scene == vision.SCENES['CHARACTER_SELECT']:
            self.set_AI_states(AI_STATES['SELECT_CHARACTER'])
        elif scene == vision.SCENES['FIGHT_PROMPT']:
            self.in_fight = True
            self.set_AI_states(AI_STATES['FIGHTING'])

    def grab_fight_info(self, img: Image.Image) -> tuple:
        p1_health, p2_health = vision.get_health_bars(img)
        info = [[p1_health], [p2_health]]
        if p1_health <= 0 or p2_health <= 0:
            self.set_AI_states(AI_STATES['INTERMISSION'])
            self.in_fight = False
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
        return(info)

    def save_tagged_img(self, img: Image.Image, info: str):
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
        player_sequence = list(range(len(self.ai_players)))
        while True:
            for i in range(len(self.key_queue)):
                self.game._dosbox.keystroke(self.key_queue.popleft())
                # TODO Clear queue once fight starts
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
