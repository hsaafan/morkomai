import os
import time
from math import floor
import pickle
from os.path import abspath

import numpy as np
from PIL import Image

from .recorder import convert_to_floats, convert_to_array, images_similar
from .recorder import open_image_array
from .globals import *


class Sprite:
    """Class for sprite images.

    Attributes
    ----------
    data: np.ndarray
        The image data of shape (height, width, channels).
    w: int
        The sprite width in px.
    h: int
        The sprite height in px.
    channels: int
        The number of sprite channels.
    char: int
        The character id corresponding to the sprite.
    desc: str
        A description of the sprite.
    mask: np.ndarray
        A mask that is True for all non-blank pixels and False otherwise.
        A blank pixel is one which has a value of 0 in all channels.
    non_blanks: int
        The number of non blank pixels in the sprite.
    channel_match: int
        The channel to use for sprite matching.
    colors: list
        A list of tuples with the first value in each tuple containing
        a color array, count of how many times the color appears
        in the sprite, the percentage of the sprite (excluding blank
        pixels) that the color covers and the locations of the color on the
        sprite.
    id: int
        The id of the sprite in the SpriteDB class.

    Parameters
    ----------
    img: PIL.Image.Image | numpy.ndarray
        The sprite image.
    char: int
        The character id corresponding to the sprite.
    desc: str, optional
        A description of the sprite, defaults to an empty string.
    """
    def __init__(self, img: Image.Image, char: int, desc: str = '') -> None:
        if isinstance(img, np.ndarray):
            self.data = img
        else:
            self.data = convert_to_array(img)
        self.h, self.w, self.channels = self.data.shape
        self.char = char
        self.desc = desc
        mask = np.ones((self.h, self.w), bool)
        blanks = np.all(self.data == 0, axis=-1)
        mask[blanks] = False
        self.mask = mask
        self.non_blanks = np.sum(self.mask)
        self.extract_palette()
        self.id = -1

    def extract_palette(self) -> None:
        """Store the colors and percent of sprite they take up."""
        masked_data = self.data[self.mask].reshape((-1, self.channels))
        colors, counts = np.unique(masked_data, return_counts=True, axis=0)
        # Find the channel with the most unique values and use that to attempt
        # sprite matching
        channel_uniques = 0
        channel_match = 0
        for c in range(self.channels):
            uniques = np.unique(colors[:, c]).size
            if uniques > channel_uniques:
                channel_match = c
                channel_uniques = uniques
        self.channel_match = channel_match
        p_color = counts / self.non_blanks
        self.colors = []
        for col, count, p in zip(colors, counts, p_color):
            col_locs = np.nonzero(np.all(self.data == col, axis=-1))
            self.colors.append((col, count, p, col_locs))


class SpriteDB:
    """A sprite database that should be used to store all sprite objects.

    This class should not be instanced as an object. All methods should be
    called directly from the class.

    Class Atributes
    ---------------
    sprites: list
        A list containing 1 list of sprite objects for each character id.
        Indices are character > sprite_id.
    palettes: list
        A list containing 1 list of colors occurring in all sprites of a
        character.
        Indices are character > color_index.
    color_counts: list
        A list corresponding to the palettes attribute that stores the number
        of times each color appears in all sprites.
        Indices are character > color_index
    p_color: list
        A list corresponding to the palettes attribute that stores the percent
        of the sprite that each color covers (excluding blank pixels).
        Indices are character > color_index > sprite_id
    """
    sprites = [[] for _ in range(len(CHARACTER_IDS))]
    palettes = [[] for _ in range(len(CHARACTER_IDS))]
    color_counts = [[] for _ in range(len(CHARACTER_IDS))]
    p_color = [[] for _ in range(len(CHARACTER_IDS))]

    def add_sprite(sprite: Sprite) -> None:
        """Add a new sprite to the class attributes.

        Parameters
        ----------
        sprite: Sprite
            The sprite to add.
        """
        SpriteDB.sprites[sprite.char].append(sprite)
        sprite.id = len(SpriteDB.sprites[sprite.char]) - 1
        SpriteDB._extend_palette(sprite)

    def _extend_palette(sprite: Sprite) -> None:
        """Adds the sprite colors to the palettes, color_counts and p_color
        attributes.

        Parameters
        ----------
        sprite: Sprite
            The sprite object to extract the colors from.
        """
        char = sprite.char
        for color, count, p_color, _ in sprite.colors:
            s_count = (sprite.id, p_color)
            # Find color in the character palette, if its not there, add it
            try:
                col_index = np.where(np.all(SpriteDB.palettes[char] == color,
                                            axis=-1))[0][0]
            except IndexError:
                SpriteDB.palettes[char].append(color)
                col_index = len(SpriteDB.palettes[char]) - 1
                SpriteDB.p_color[char].append([])
                SpriteDB.color_counts[char].append(0)
            SpriteDB.color_counts[char][col_index] += count
            # Add the p_color while keeping list sorted in descending order
            i = 0
            while True:
                if i >= len(SpriteDB.p_color[char][col_index]):
                    SpriteDB.p_color[char][col_index].append(s_count)
                    break
                elif SpriteDB.p_color[char][col_index][i][1] < p_color:
                    SpriteDB.p_color[char][col_index].insert(i, s_count)
                    break
                else:
                    i += 1


class SpriteMatcher:
    """Used to match pixels on screen to sprites.

    Attributes
    ----------
    timeout: float
        Seconds to wait per player before halting search.
    env_palette: list
        A palette containing a list of colors used in the scene. These are only
        the colors that appear on the edges behind the players. This is to
        prevent taking any extra colors which might mess with the matching.
    img: np.ndarray
        The image to match sprites to. Shape is (height, width, channels).

    Parameters
    ----------
    p1_char: int
        The character id of the first player. See CHARACTER_IDS.
    p2_char: int
        The character id of the second player. See CHARACTER_IDS.
    ppr: int, optional
        Number of samples to take from each row, defaults to 45.
    ppc: int, optional
        Number of samples to take from each column, defaults to 15.
    timeout: float, optional
        Seconds to wait per player before halting search, defaults to 0.15.
    """
    def __init__(self, p1_char: int, p2_char: int,
                 ppr: int = 45, ppc: int = 15,
                 timeout: float = 0.15) -> None:
        self._ppr = ppr
        self._ppc = ppc
        self.timeout = timeout
        self.env_palette = []
        self.change_chars(p1_char, p2_char)

    def update_image(self, img: Image.Image) -> None:
        """Update the stored image.

        Parameters
        ----------
        img: PIL.Image.Image
            The image to store.
        """
        self.img = convert_to_array(img)

    def change_chars(self, p1_char: int, p2_char: int) -> None:
        """Change character ids.

        Parameters
        ----------
        p1_char: int
            The character id of the first player.
        p2_char: int
            The character id of the second player.
        """
        self._p1_char = p1_char
        if p1_char == p2_char:
            self._p2_char = CHARACTER_IDS[f'{CHARACTERS[p2_char]} (Alternate)']
        else:
            self._p2_char = p2_char

    def find_players(self, bounds: tuple = None,
                     only_p1: bool = False, only_p2: bool = False) -> list:
        """Find the player sprites on screen.

        Parameters
        ----------
        bounds: tuple, optional
            Bounding box where pixels are sampled from. The order of the tuple
            is x_left, y_top, x_right, y_bottom. If None, will sample the whole
            screen.
        only_p1: bool, optional
            Only look for the first player, defaults to False.
        only_p2: bool, optional
            Only look for the second player, defaults to False.

        Returns
        -------
        bounding_boxes: list
            A list of the bounding boxes where the sprites were found. Each
            item in this list is a tuple where the first entry is the id of the
            sprite and the second entry is a tuple describing the bounding box.
            The bounding box tuple contains x_left, y_top, x_right, y_bottom.
        """
        if not only_p1 and not only_p2:
            # Search the whole image for both players
            bounding_boxes = [self.find_players(bounds, only_p1=True)[0],
                              self.find_players(bounds, only_p2=True)[0]]
        else:
            if bounds is None:
                bounded_img = self.img
            else:
                l, t, r, b = bounds
                bounded_img = self.img[t:b, l:r]
            pixels = sample_pixels(bounded_img, self._ppr, self._ppc, l, t)
            mid = int(len(pixels) / 2)
            pixels = pixels[mid:] + pixels[:mid]

            if only_p1:
                bounding_boxes = [self.match(pixels, self._p1_char)]
            elif only_p2:
                bounding_boxes = [self.match(pixels, self._p2_char)]
        return(bounding_boxes)

    def match(self, pixels: list, char: int, col_tol: int = 5) -> tuple:
        """Match one of the character sprites to a location in the image.

        Parameters
        ----------
        pixels: list
            The list of pixels in the image to look through. Each pixel in the
            list is a list like object with the x, y coordinates in the first
            two positions and a numpy ndarray describing the color of the pixel
            in the third position.
        char: int
            The character id whose sprites are being checked.
        col_tol: int, optional
            The tolerance of the color when matching the pixels. Defaults to 5.
            For a color [220, 100, 85] with a tolerance of 5, matches are any
            colors between [215, 195, 80] and [225, 105, 90].

        Returns
        -------
        match: tuple | None
            Returns None if no match found, otherwise returns a tuple with the
            first value equal to the sprite id that was matched and the second
            value equal to the bounding box of the sprite. The bounding box is
            a tuple with values x_left, y_top, x_right, y_bottom.
        """
        pixel_order = []
        min_x, min_y, max_x, max_y = (1e6, 1e6, 0, 0)
        for x, y, color in pixels:
            # Order sampled pixels by how rare they are in the character
            # sprites
            if np.all(color == 0):
                continue  # Dont match empty spots (black in RGB)
            if color_in_palette(color, self.env_palette, col_tol) is not None:
                continue  # Dont match pixels in environment palette
            c = color_in_palette(color, SpriteDB.palettes[char], col_tol)
            if c is None:
                continue
            count = SpriteDB.color_counts[char][c]
            new_pixel = (x, y, c)
            if x < min_x:
                min_x = x
            elif x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            elif y > max_y:
                max_y = y
            i = 0
            while True:
                if i >= len(pixel_order):
                    pixel_order.append(new_pixel)
                    break
                elif SpriteDB.color_counts[char][pixel_order[i][2]] > count:
                    pixel_order.insert(i, new_pixel)
                    break
                else:
                    i += 1
        min_size = (max_x - min_x, max_y - min_y)
        match = search_pixels(self.img, pixel_order, char,
                              self.timeout, min_size)
        return(match)

    def update_environment_palette(self) -> None:
        """Updates the environment palette."""
        self.env_palette = get_environment_palette(self.img)


def load_sprite(filename: str, char: int, desc:  str = '') -> None:
    """Load a sprite from a file and store it in SpriteDB class.

    Parameters
    ----------
    filename: str
        The path of the sprite to open.
    char: int
        The character id corresponding to the sprite.
    desc: str, optional
        A description of the sprite, defaults to an empty string.
    """
    sprite_img = Image.open(filename)
    flipped_img = sprite_img.transpose(Image.FLIP_LEFT_RIGHT)
    sprite = Sprite(sprite_img, char, desc)
    flipped_sprite = Sprite(flipped_img, char, desc + '(R)')
    SpriteDB.add_sprite(sprite)
    SpriteDB.add_sprite(flipped_sprite)


def load_sprite_folder(folder: str, char: int) -> None:
    """Load all sprites from a folder (non-recusrive).

    The description of the sprites is set as the filename (without the
    extension). Only png files are loaded.

    Parameters
    ----------
    folder: str
        The path of the folder.
    char: int
        The character id corresponding to the sprites in the folder.
    """
    directory = os.fsencode(folder)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            full_path = os.path.join(folder, filename)
            load_sprite(full_path, char, filename[:-4])


def load_all_sprites(root_path: str = 'images/sprites',
                     quiet: bool = False) -> None:
    """Loads all sprites (naming must be default convention).

    Parameters
    ----------
    root_path: str
        The root folder that contains all sprite folders.
    quiet: bool
        If False will print a message when called. Defaults to False.
    """
    if not quiet:
        print('Loading and preprocessing sprites.')
    names = ['Cage', 'Kano', 'Raiden', 'Kang', 'Scorpion', 'Subzero', 'Sonya']
    folders = []
    for x in names:
        folders.append(f'{root_path}/{x}')
    folders.append(f'{root_path}/Goro')
    folders.append(f'{root_path}/Tsung')
    for x in names:
        folders.append(f'{root_path}/{x}/AltSkin')
    for i, folder_path in enumerate(folders):
        load_sprite_folder(folder_path, i)
    load_shang_tsung()


def load_shang_tsung() -> None:
    """Copies all character sprites to Shang Tsung.

    Includes all playbale character sprites into Shang Tsungs SpriteDB entry
    since he can transform into them. This must only be called after loading
    all other sprites.
    """
    for char, chard_id in CHARACTER_IDS.items():
        if char == 'Goro' or char == 'Shang Tsung':
            continue
        for sprite in SpriteDB.sprites[chard_id]:
            img = sprite.data
            desc = sprite.desc
            shang_sprite = Sprite(img, CHARACTER_IDS['Shang Tsung'], desc)
            SpriteDB.add_sprite(shang_sprite)


def build_spritedb(root_path: str = 'images/sprites') -> None:
    """Loads sprites from disk and pickles the processed data.

    Parameters
    ----------
    root_path: str
        The root folder that contains all sprite folders.
    """
    load_all_sprites(root_path)
    with open(f'{abspath(root_path)}/{DB_PICKLE_FILENAME}', 'wb') as f:
        db_data = {
            'sprites': SpriteDB.sprites,
            'palettes': SpriteDB.palettes,
            'color_counts': SpriteDB.color_counts,
            'p_color': SpriteDB.p_color
        }
        pickle.dump(db_data, f)


def load_spritedb(root_path: str = 'images/sprites') -> None:
    """Load preprocessed sprite data or if not available, raw images from disk.

    Parameters
    ----------
    root_path: str
        The root folder that contains all sprite folders.
    """
    try:
        with open(f'{abspath(root_path)}/{DB_PICKLE_FILENAME}', 'rb') as f:
            db_data = pickle.load(f)
        SpriteDB.sprites = db_data['sprites']
        SpriteDB.palettes = db_data['palettes']
        SpriteDB.color_counts = db_data['color_counts']
        SpriteDB.p_color = db_data['p_color']
    except FileNotFoundError:
        build_spritedb(root_path)


def detect_current_scene(scene_img: Image.Image,
                         join_screen: np.ndarray,
                         char_select: np.ndarray,
                         continue_screen: np.ndarray,
                         fight_prompt: np.ndarray) -> int:
    """Match an image to a preset scene.

    Parameters
    ----------
    scene_img: PIL.Image.Image
        The image to match.
    join_screen: np.ndarray
        The startup screen at which players can start joining. It should be an
        np.ndarray of floats.
    char_select: np.ndarray
        The character select screen. It should be an np.ndarray of floats.
    continue_screen: np.ndarray
        The continue screen after losing a fight. It should be an np.ndarray
        of floats.
    fight_prompt: np.ndarray
        An image containing only the fight prompt at where it should appear on
        screen. The image should be an np.ndarray of ints in [0, 255].

    Returns
    -------
    scene: int
        The scene that was detected. See SCENES.
    """
    c_pos = CONTINUE_POSITION
    if at_fight_prompt(convert_to_array(scene_img), fight_prompt):
        scene = SCENES['FIGHT_PROMPT']
    else:
        scene_array = convert_to_floats(scene_img)
        if images_similar(scene_array, join_screen, 0.1):
            scene = SCENES['INTRODUCTION']
        elif images_similar(scene_array, char_select, 0.1):
            scene = SCENES['CHARACTER_SELECT']
        elif images_similar(scene_array[c_pos[1]:c_pos[3], c_pos[0]:c_pos[2]],
                            continue_screen, 0.1):
            scene = SCENES['INTRODUCTION']  # Same actions needed as in intro
        else:
            scene = SCENES['OTHER']
    return(scene)


def at_fight_prompt(scene: np.ndarray, fight_prompt: np.ndarray) -> bool:
    """Is there a fight prompt on the current screen.

    Parameters
    ----------
    scene: np.ndarray
        The image to match. The image should be an np.ndarray of ints in
        [0, 255].
    fight_prompt: np.ndarray
        An image containing only the first channel of the fight prompt at where
        it should appear on screen. The image should be an np.ndarray of floats
        in [0, 1].

    Returns
    -------
    at_fight_prompt: bool
        Is there a fight prompt on the current screen.
    """
    indices = np.ceil(fight_prompt).astype(bool)  # Only non-blank pixels
    red_channel = scene[:, :, 0] / 255
    return(images_similar(fight_prompt[indices], red_channel[indices]))


def get_health_bars(scene: np.ndarray) -> tuple:
    """Get the health bar images from the scene.

    Parameters
    ----------
    scene: np.ndarray
        The current fight scene. The image should be an np.ndarray of ints in
        [0, 255].

    Returns
    -------
    p1_bar: np.ndarray
        The health bar of the first player.
    p2_bar: np.ndarray
        The health bar of the second player.
    """
    p1_pos, p2_pos = HEALTH_BAR_POSITIONS
    p1_bar = scene[p1_pos[1]:p1_pos[3], p1_pos[0]:p1_pos[2]]
    p2_bar = scene[p2_pos[1]:p2_pos[3], p2_pos[0]:p2_pos[2]]
    return(p1_bar, p2_bar)


def detect_character(scene: np.ndarray, player: int,
                     nameplate_folder: str = 'images/names') -> int:
    """Detect the character based on the health bar name.

    Parameters
    ----------
    scene: np.ndarray
        The current fight scene. The image should be an np.ndarray of ints in
        [0, 255].
    player: int
        Which player to detect the character for.
    nameplate_folder: str
        The folder containing the nameplates. Defaults to 'images/names'.

    Returns
    -------
    character: str
        The detected character. See CHARACTERS.
    """
    health_bar = get_health_bars(scene)[player][:, :, 1]
    _, bar_length = health_bar.shape

    directory = os.fsencode(nameplate_folder)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            full_path = os.path.join(nameplate_folder, filename)
            character_name = filename[:-4]

            nameplate = open_image_array(full_path)[:, :, 1]  # Only g channel
            _, width = nameplate.shape
            indices = nameplate > 170  # Non blank pixels

            for x in range(bar_length):
                try:
                    if images_similar(health_bar[:, x:x+width][indices],
                                      nameplate[indices], 0.1):
                        return(character_name)
                except IndexError:
                    break
    raise RuntimeError('Could not detect character from nameplate.')


def detect_character_select_positions(scene: np.ndarray) -> int:
    """Detect the cursor on the character select screen.

    Parameters
    ----------
    scene: np.ndarray
        The current scene. The image should be an np.ndarray of ints in
        [0, 255].

    Returns
    -------
    p1_position: int | None
        The position of player 1 on the character select screen.
    p2_position: int | None
        The position of player 2 on the character select screen.
    """
    p1_positon, p2_position = (None, None)
    for key, (x1, y1, x2, y2) in CHARACTER_SELECT_FRAMES.items():
        frame_mask = np.ones((y2 - y1, x2 - x1), dtype=bool)
        # Cursor frame is 3 pixels thick
        frame_mask[3:-3, 3:-3] = False
        frame_size = frame_mask.sum()
        frame = scene[y1:y2, x1:x2]
        green_frame = (frame[frame_mask, 1] > 100).sum() / frame_size > 0.9
        red_frame = (frame[frame_mask, 0] > 100).sum() / frame_size > 0.9
        if green_frame and not red_frame:
            # Green cursor for player 1
            p1_positon = CHARACTER_IDS[key]
        elif red_frame and not green_frame:
            # Red cursor for player 2
            p2_position = CHARACTER_IDS[key]
    return(p1_positon, p2_position)


def is_timer_done(scene: np.ndarray, timer_image: np.ndarray) -> bool:
    """Detect whether the time reads 00.

    Parameters
    ----------
    scene: np.ndarray
        The current scene. The image should be an np.ndarray of ints in
        [0, 255].

    Returns
    -------
    is_timer_done: bool
        True if the timer reads 00 and False otherwise.
    """
    timer = scene[TIMER_POSITION[1]:TIMER_POSITION[3],
                  TIMER_POSITION[0]:TIMER_POSITION[2], 0]
    mask = timer_image != 0
    return(images_similar(timer[mask], timer_image[mask]))


def get_health(scene: np.ndarray) -> tuple:
    """Get the current health of characters based on an image of the scene.

    Parameters
    ----------
    scene: np.ndarray
        The current fight scene. The image should be an np.ndarray of ints in
        [0, 255].

    Returns
    -------
    p1_health: float
        Health of first player in [0, 1].
    p2_health: float
        Health of second player in [0, 1].
    """
    p1_bar, p2_bar = get_health_bars(scene)
    p1_bar = np.flip(p1_bar[:, :, 1])
    p2_bar = p2_bar[:, :, 1]

    bar_height, bar_length = p1_bar.shape
    p1_first, p2_first = (bar_length, bar_length)
    for i in range(bar_height):
        # Iterate through rows and find first green pixels
        # Goes through all rows since name can cover up part of the bar
        p1_pixel = next((idx for idx, g in np.ndenumerate(p1_bar[i])
                         if 160 < g < 170), (bar_length, ))
        p2_pixel = next((idx for idx, g in np.ndenumerate(p2_bar[i])
                         if 160 < g < 170), (bar_length, ))

        if p1_pixel[0] < p1_first:
            p1_first = p1_pixel[0]
        if p2_pixel[0] < p2_first:
            p2_first = p2_pixel[0]

    p1_health = (bar_length - p1_first) / bar_length
    p2_health = (bar_length - p2_first) / bar_length
    return(p1_health, p2_health)


def sample_pixels(array: np.ndarray, ppr: int, ppc: int,
                  x_offset: int = 0, y_offset: int = 0) -> list:
    """Return a uniform set of values in an array.

    Parameters
    ----------
    array: np.ndarray
        The array to sample from.
    ppr: int
        Number of samples to take per row.
    ppc: int
        Number of samples to take per column.
    x_offset: int, optional
        Adds an offset to x coordinates in returned list. Defaults to 0.
    y_offset: int, optional
        Adds an offset to y coordinates in returned list. Defaults to 0.

    Returns
    -------
    samples: list
        A list containing the samples, each item in the list is a tuple made
        of (x + x_offset, y + y_offset, array[y, x]).
    """
    height = array.shape[0]
    width = array.shape[1]
    x_gap = floor(width / (ppr + 1))
    y_gap = floor(height / (ppc + 1))
    row_indices = list(range(0, height, y_gap))
    col_indices = list(range(0, width, x_gap))
    samples = [(x + x_offset, y + y_offset, array[y, x])
               for x in col_indices for y in row_indices]
    return(samples)


def find_color_matches(image: np.ndarray, color: np.ndarray,
                       col_tol: int = 5) -> np.ndarray:
    """Finds all pixels on image that matches a color within a tolerance.

    Parameters
    ----------
    image: np.ndarray
        The image to search through.
    color: np.ndarray
        The color to match to.
    col_tol: int, optional
        The tolerance of the color when matching the pixels. Defaults to 5.
        For a color [220, 100, 85] with a tolerance of 5, matches are any
        colors between [215, 195, 80] and [225, 105, 90].

    Returns
    -------
    coords: tuple
        A tuple of indices where matches occur. The size of the tuple is
        equal to 1 less than the number of dimensions of the image.
    """
    above = (image >= color - col_tol).all(axis=-1)
    below = (image <= color + col_tol).all(axis=-1)
    coords = np.nonzero(above * below)
    return(coords)


def search_pixels(img: np.ndarray, pixels: list, char: int,
                  timeout: float = 0.25,
                  min_size: tuple = None,
                  min_size_pcnt: float = 0.5) -> tuple:
    """Search through a set of pixels in order and match to a sprite.

    Parameters
    ----------
    img: np.ndarray
        The image to search through.
    pixels: list
        A list of tuples each containing an x coordinate, y coordinate, and
        a color index.
    char: int
        The character whose sprites to search through. See CHARACTER_IDS.
    timeout: float, optional
        The maximum time in seconds to search. Defaults to 0.25.
    min_size: tuple, optional
        A tuple containing the minimum width and height of sprites to search
        through. Defaults to None which searches all sprites.
    min_size_pcnt: float, optional
        A modifier on the min_size tuple values. Defaults to 0.5.

    Returns
    -------
    match: tuple | None
        The sprite that was matched. The first value in the tuple is the sprite
        id while the second is a tuple of the bounding box (x1, y1, x2, y2).
        Returns None if no match was found.
    """
    start = time.time()
    for x, y, c in pixels:
        sprite_p_color = SpriteDB.p_color[char][c]
        for k, _ in sprite_p_color:
            # Search in descending order of most color coverage in sprite
            sprite = SpriteDB.sprites[char][k]
            if min_size is not None:
                if sprite.w < min_size_pcnt * min_size[0]:
                    continue
                elif sprite.h < min_size_pcnt * min_size[1]:
                    continue
            time_left = timeout - (time.time() - start)
            sprite_match = find_img(img, (x, y), sprite, timeout=time_left)
            if sprite_match is not None:
                return(sprite_match)
            if time.time() - start > timeout:
                break


def find_img(img: np.ndarray, coords: tuple, sprite: Sprite,
             min_match: float = 0.8,
             col_tol: int = 5,
             timeout: float = 0.5) -> tuple:
    """Attempt to match a sprite to a location on an image.

    The match starts at the given coordinates on the image and iterates
    through the pixels on the sprite that match the color of the pixel. The
    match box grows with each iteration until either the min_match is not
    met, or the full sprite is matched.

    Parameters
    ----------
    img: np.ndarray
        The image to find the sprite in.
    coords: tuple
        The image pixel coordinates to match to. Values are (x, y).
    sprite: Sprite
        Sprite to attempt match on.
    min_match: float, optional
        The percent of pixels that need to be matched for the sprite to
        match to the image. Defaults to 0.8. Setting this too low will
        increase search time but setting it to high will not match when
        characters are covered.
    col_tol: int, optional
        The tolerance of the color when matching the pixels. Defaults to 5.
        For a color [220, 100, 85] with a tolerance of 5, matches are any
        colors between [215, 195, 80] and [225, 105, 90].
    timeout: float, optional
        The maximum time in seconds to search. Defaults to 0.25.

    Returns
    -------
    match: tuple | None
        The bounding box on the image where the sprite is matched to. The
        tuple contains the values x_left, y_top, x_right, y_bottom. If no
        match is found, returns None.
    """
    sprite_h, sprite_w, sprite_channels = sprite.data.shape
    _, _, channels = img.shape
    if sprite_channels != channels:
        raise TypeError('Number of channels do not match')
    channel_match = sprite.channel_match

    start_time = time.time()
    x, y = coords
    max_height_to_search = PLAY_AREA[3] - y
    for c, _, _, match_coords in sprite.colors:
        above = (img[y, x] >= c - col_tol).all()
        below = (img[y, x] <= c + col_tol).all()
        if above and below:
            break
    sprite_x1, sprite_y1, sprite_x2, sprite_y2 = (0, 0, 0, 0)
    crop_x1, crop_y1, crop_x2, crop_y2 = (0, 0, 0, 0)

    for match_y, match_x in zip(match_coords[0], match_coords[1]):
        if time.time() - start_time > timeout:
            break
        if (sprite_h - match_y) > max_height_to_search:
            # Don't attempt to match coords of sprite if the match would
            # mean the sprite clips below the play area
            continue
        rad = 1
        prev_tiles = 0
        prev_matches = 0
        # Maximum radius to cover image
        target_radius = sprite_w - match_x
        if sprite_h - match_y > target_radius:
            target_radius = sprite_h - match_y
        if match_x > target_radius:
            target_radius = match_x
        if match_y > target_radius:
            target_radius = match_y
        # Crop the image to what the sprite bounding box would be
        crop_x1 = x - match_x
        crop_y1 = y - match_y
        crop_x2 = crop_x1 + sprite_w
        crop_y2 = crop_y1 + sprite_h
        cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        psprite_x1 = None  # No previous check applied on sprite

        match_critera = 0.1
        while True:
            match_critera *= 1.5
            if match_critera > min_match:
                match_critera = min_match
            # Sprite coordinates
            sprite_x1 = match_x - rad
            sprite_y1 = match_y - rad
            sprite_x2 = match_x + rad
            sprite_y2 = match_y + rad
            if sprite_x1 < 0:
                sprite_x1 = 0
            if sprite_y1 < 0:
                sprite_y1 = 0
            if sprite_x2 > sprite_w:
                sprite_x2 = sprite_w
            if sprite_y2 > sprite_h:
                sprite_y2 = sprite_h

            # Get a cropped view of the sprite and image to compare
            sprite_view = sprite.data[sprite_y1:sprite_y2, sprite_x1:sprite_x2,
                                      channel_match]
            image_view = cropped_img[sprite_y1:sprite_y2, sprite_x1:sprite_x2,
                                     channel_match]
            mask = sprite.mask[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
            # Mask all previous checked entries
            if psprite_x1 is not None:
                prev_l = psprite_x1 - sprite_x1
                prev_t = psprite_y1 - sprite_y1
                prev_r = sprite_x2 - psprite_x2
                prev_b = sprite_y2 - psprite_y2
                mask[prev_t:prev_b, prev_l:prev_r] = False
            sprite_view = sprite_view[mask]
            image_view = image_view[mask]
            # Number of non-blank tiles checked so far
            n_tiles = sprite_view.size + prev_tiles
            # Compare
            diff = np.abs(sprite_view - image_view)
            # np.ndarray.co
            new_matches = np.count_nonzero(diff < col_tol)
            matches = prev_matches + new_matches
            try:
                match_pcnt = matches / n_tiles
            except ZeroDivisionError:
                match_pcnt = 1
            if match_pcnt < match_critera:
                break  # Sprite doesn't match here, go to next coordinate
            elif rad >= target_radius:
                # The whole image has been checked and matches
                return(sprite.id, (x - match_x,
                                   y - match_y,
                                   x - match_x + sprite_w,
                                   y - match_y + sprite_h))
            # Increase radius and store previous check values
            rad *= 2
            psprite_x1 = sprite_x1
            psprite_y1 = sprite_y1
            psprite_x2 = sprite_x2
            psprite_y2 = sprite_y2
            prev_matches = matches
            prev_tiles = n_tiles
    return(None)  # No matches on sprite


def get_environment_palette(img: np.ndarray) -> list:
    """Extracts the palette of the environment excluding the characters.

    Parameters
    ----------
    img: np.ndarray
        The scene image.

    Returns
    -------
    palette: list
        The list of colors used in the scene.
    """
    left = img[PLAY_AREA[1]:PLAY_AREA[3], PLAY_AREA[0]:START_POSITIONS[0][0]]
    right = img[PLAY_AREA[1]:PLAY_AREA[3], START_POSITIONS[1][2]:PLAY_AREA[2]]
    joined = np.hstack((left, right)).reshape((-1, left.shape[2]))
    palette = np.unique(joined, axis=0)
    return(list(palette))


def color_in_palette(color: np.ndarray, palette: list,
                     col_tol: int = 5) -> int:
    """Returns index of color if color is in palette.

    Parameters
    ----------
    color: np.ndarray
        The color to find.
    palette: int
        The palette to search through.
    col_tol: int
        The tolerance of each color channel.
    """
    try:
        return(find_color_matches(palette, color, col_tol)[0][0])
    except IndexError:
        return(None)  # Color doesn't match any in character palette
