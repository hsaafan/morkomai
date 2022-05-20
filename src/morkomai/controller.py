import threading
from .mortalkombat import MortalKombat
from .ai import AI
from . import vision
from .vision import SpriteMatcher, load_sprite_folder, START_POSITIONS
from PIL import ImageDraw
import time
import numpy as np


def tag_sprites(game: MortalKombat):
    matcher = SpriteMatcher(0, 1, 10, 10, timeout=1)
    bounds = START_POSITIONS[0]
    text = 'idle1'
    i = 0
    while True:
        img = game._recorder.get_current_image()
        matcher.update_image(img)
        try:
            match = matcher.find_players(only_p1=True, bounds=bounds)[0]
        except IndexError:
            match = None
        if match is not None:
            text = vision.SpriteDB.sprites[0][match[0]].desc
            bounds = match[1]
        draw = ImageDraw.Draw(img)
        draw.rectangle(list(bounds), outline="red", width=3)
        draw.text((bounds[0], bounds[1] - 5), text)
        img.save(f'.captures/{i:05d}.png')
        i += 1
        bounds = [np.max([bounds[0] - 20, 160]), np.max([bounds[1] - 5, 140]),
                  np.min([bounds[2] + 20, 480]), np.min([bounds[3] + 5, 340])]


def start_game(AI_1: bool = False, AI_2: bool = True,
               AI_1_tactics: str = 'random',
               AI_2_tactics: str = 'random') -> MortalKombat:
    threads = []
    load_sprite_folder('images/sprites/Cage', 0)
    mk1 = MortalKombat()
    mk1.start()
    AI_players = []

    if AI_1:
        AI_players.append((AI(mk1, 0, reporting=True), AI_1_tactics))
    if AI_2:
        AI_players.append((AI(mk1, 1, reporting=True), AI_2_tactics))

    for player_tuple in AI_players:
        player, tactics = player_tuple
        if tactics == 'random':
            player.fight = player.random_moves
        threads.append(threading.Thread(target=player.run))
    sprite_thread = threading.Thread(target=tag_sprites, kwargs={'game': mk1})

    try:
        for player_thread in threads:
            player_thread.start()
        time.sleep(15)
        sprite_thread.start()
    except ChildProcessError:
        print('\ndosbox or the display server have stopped')
