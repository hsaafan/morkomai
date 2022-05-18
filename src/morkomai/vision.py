import numpy as np
from math import floor
from PIL import Image
import os


def sample_pixels(array: np.ndarray, ppr: int, ppc: int) -> list:
    """Return a uniform set of values in an array.

    Parameters
    ----------
    array: np.ndarray
        The array to sample from.
    ppr: int
        Number of rows to take samples from.
    ppc: int
        Number of columns to take samples from.
    """
    height = array.shape[0]
    width = array.shape[1]
    x_gap = floor(width / (ppr + 1))
    y_gap = floor(height / (ppc + 1))

    samples = []
    x, y = (0, 0)
    while True:
        x += x_gap
        if x >= width:
            break
        while True:
            y += y_gap
            if y >= height:
                y = 0
                break
            samples.append([x, y, array[y, x]])
    return(samples)


class SpriteMatcher:
    def __init__(self, p1_char: int, p2_char: int,
                 ppr: int = 3, ppc: int = 3) -> None:
        """Used to match pixels on screen to sprites.

        Parameters
        ----------
        p1_char: int
            The character id of the first player.
        p2_char: int
            The character id of the second player.
        ppr: int, optional
            Number of rows to take samples from, defaults to 3.
        ppc: int, optional
            Number of columns to take samples from, defaults to 3.

        Attributes
        ----------
        img: np.ndarray
            The image to match sprites to. Shape is (height, width, channels).
        """
        self._ppr = ppr
        self._ppc = ppc
        self.change_chars(p1_char, p2_char)

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
            self._p2_char = p2_char + 7
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
            pixels = sample_pixels(bounded_img, self._ppr, self._ppc)

            if bounds is not None:
                for i, (x, y, _) in enumerate(pixels):
                    # Remap pixels to the full image
                    pixels[i][0] = x + l
                    pixels[i][1] = y + t

            if only_p1:
                bounding_boxes = [self.match(self.img, pixels, self._p1_char)]
            elif only_p2:
                bounding_boxes = [self.match(self.img, pixels, self._p2_char)]
        return(bounding_boxes)

    def match(self, img: np.ndarray, pixels: list, char: int,
              color_tol: int = 5) -> tuple:
        """Match one of the character sprites to a location in the image.

        Parameters
        ----------
        img: np.ndarray
            The image to look through. Shape is (height, width, channels).
        pixels: list
            The list of pixels in the image to look through. Each pixel in the
            list is a list like object with the x, y coordinates in the first
            two positions and a numpy ndarray describing the color of the pixel
            in the third position.
        char: int
            The character id whose sprites are being checked.
        color_tol: int, optional
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
        for x, y, color in pixels:
            if np.all(color == np.zeros_like(color)):
                continue  # Dont match empty spots (black in RGB)
            color_index = -1
            for i, char_color in enumerate(SpriteDB.palettes[char]):
                # Check if pixel color is in character palette
                above_min_color = np.all((color - color_tol) <= char_color)
                below_max_color = np.all(char_color <= (color + color_tol))
                if above_min_color and below_max_color:
                    color_index = i
                    break
            if color_index == -1:
                continue  # Color doesn't match any in character palette
            sprite_color_counts = SpriteDB.color_counts[char][color_index]
            for i, _ in sprite_color_counts:
                # Search in descending order of most color appearances
                sprite = SpriteDB.sprites[char][i].data
                sprite_match = self.find_img(img, (x, y), sprite)
                if sprite_match is not None:
                    return(i, sprite_match)
        return(None)

    def find_img(self, img: np.ndarray, coords: tuple, sprite: np.ndarray,
                 min_match: float = 0.33, color_tol: int = 5) -> tuple:
        """Attempt to match a sprite to a location on an image.

        The match starts at the given coordinates on the image and iterates
        through the pixels on the sprite that match the color of the pixel. The
        match box grows with each iteration until either the min_match is not
        met, or the full sprite is matched.

        Parameters
        ----------
        img: np.ndarray
            Image to attempt match on with shape (height, width, channels).
        coords: tuple
            The pixel coordinates to match to. Values are (x, y).
        sprite: np.ndarray
            Sprite to attempt match on with shape (height, width, channels).
        min_match: float, optional
            The percent of pixels that need to be matched for the sprite to
            match to the image. Defaults to 0.33.
        color_tol: int, optional
            The tolerance of the color when matching the pixels. Defaults to 5.
            For a color [220, 100, 85] with a tolerance of 5, matches are any
            colors between [215, 195, 80] and [225, 105, 90].

        Returns
        -------
        match: tuple | None
            The bounding box on the image where the sprite is matched to. The
            tuple contains the values x_left, y_top, x_right, y_bottom. If no
            match is found, returns None.
        """
        sprite_h, sprite_w, sprite_channels = sprite.shape
        img_h, img_w, channels = img.shape
        if sprite_channels != channels:
            raise TypeError('Number of channels do not match')
        color_tol *= channels

        x, y = coords
        color_locs = self.find_color_matches(sprite, img[y, x])
        rad = 1
        match_coords = np.nonzero(color_locs)
        for match_y, match_x in zip(match_coords[0], match_coords[1]):
            match_h, match_w, _ = sprite.shape
            while True:
                # Sprite coordinates
                left = max([match_x - rad, 0])
                top = max([match_y - rad, 0])
                right = min([match_x + rad, sprite_w])
                bottom = min([match_y + rad, sprite_h])

                # Matching image coordinates
                img_left = x - min([match_x - left, rad])
                img_top = y - min([match_y - top, rad])
                img_right = x + min([right - match_x, rad])
                img_bottom = y + min([bottom - match_y, rad])

                # Sprite may be off screen, shrink sprite and image boxes
                # BUG Will not work if sprite stretches across whole screen,
                # ignore because this isn't possible for MK1
                if img_left < 0:
                    left -= img_left
                    match_w = sprite_w - img_left
                    img_left = 0
                if img_top < 0:
                    top -= img_top
                    match_h = sprite_h - img_top
                    img_top = 0
                if img_right > img_w:
                    right -= (img_right - img_w)
                    match_w = sprite_w - (img_right - img_w)
                    img_right = img_w
                if img_bottom > img_h:
                    bottom -= (img_bottom - img_h)
                    match_h = sprite_h - (img_bottom - img_h)
                    img_bottom = img_h

                # Compare the sprite and image
                sprite_view = sprite[top:bottom, left:right]
                img_view = img[img_top:img_bottom, img_left:img_right]
                diff = np.abs(sprite_view - img_view)
                matches = np.max(diff, axis=2) < color_tol
                match_pcnt = np.sum(matches) / (img_view.size / channels)
                if match_pcnt < min_match:
                    break  # Not enough pixels match, go to next coordinate
                elif (right - left) >= match_w and (bottom - top) >= match_h:
                    return(img_left, img_top, img_right, img_bottom)
                rad *= 2
        return(None)

    def find_color_matches(self, image: np.ndarray,
                           color: np.ndarray,
                           color_tol: int = 5) -> np.ndarray:
        """Finds all pixels on image that matches a color within a tolerance.

        Parameters
        ----------
        image: np.ndarray
            The image to search through.
        color: np.ndarray
            The color to match to.
        color_tol: int, optional
            The tolerance of the color when matching the pixels. Defaults to 5.
            For a color [220, 100, 85] with a tolerance of 5, matches are any
            colors between [215, 195, 80] and [225, 105, 90].

        Returns
        -------
        matches: np.ndarray
            An array with the same shape image except for the last axis which
            is removed. The array contains boolean values that indicate whether
            pixel colors match.
        """
        above = np.logical_and.reduce(image >= color - color_tol, axis=-1)
        below = np.logical_and.reduce(image <= color + color_tol, axis=-1)
        return(above * below)

    def update_image(self, img: Image) -> None:
        """Update the stored image.

        Parameters
        ----------
        img: Classes from PIL.Image.open
            The image to store.
        """
        width, height = img.size
        self.img = np.asarray(img.getdata()).reshape((height, width, 3))


class Sprite:
    def __init__(self, img: np.ndarray, char: int, desc: str = '') -> None:
        """Class for sprite images.

        Parameters
        ----------
        img: np.ndararay | Classes from PIL.Image.open
            The sprite image, if its a np.ndarray, its shape should be
            (height, width, channels).
        char: int
            The character id corresponding to the sprite.
        desc: str, optional
            A description of the sprite, defaults to an empty string.

        Attributes
        ----------
        data: np.ndarray
            The image data of shape (height, width, channels).
        char: int
            The character id corresponding to the sprite.
        desc: str
            A description of the sprite.
        colors: zip
            A zipped iterable with the first value containing color arrays and
            the second containing a count of how many times the color appears
            in the sprite.
        """
        if type(img) != np.ndarray:
            w, h = img.size
            img = np.asarray(img.getdata()).reshape((h, w, 3))
        if len(img.shape) != 3:
            raise TypeError('Expected a 3 channel RGB image')
        self.data = img
        self.char = char
        self.desc = desc
        self.extract_palette()

    def extract_palette(self) -> None:
        """Store the colors and number of times they appear in the sprite."""
        colors, counts = np.unique(self.data.reshape((-1, 3)),
                                   return_counts=True, axis=0)
        self.colors = zip(colors, counts)


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
        of times each color appears in a sprite.
        Indices are character > color_index > sprite_id
    """
    sprites = [[] * 16]
    palettes = [[] * 16]
    color_counts = [[] * 16]

    def add_sprite(sprite: Sprite) -> int:
        """Add a new sprite to the class attributes

        Parameters
        ----------
        sprite: Sprite
            The sprite to add.

        Returns
        -------
        sprite_id: int
            The id of the sprite that was added.
        """
        SpriteDB.sprites[sprite.char].append(sprite)
        sprite_id = len(SpriteDB.sprites[sprite.char]) - 1
        SpriteDB._extend_palette(sprite, sprite_id)
        return(sprite_id)

    def _extend_palette(sprite: Sprite, sprite_id: int) -> None:
        """Adds the sprite colors to the palettes and color_counts attributes

        Parameters
        ----------
        sprite: Sprite
            The sprite object to extract the colors from.
        sprite_id: int
            The id of the sprite to extract the colors from.
        """
        char = sprite.char
        for color, count in sprite.colors:
            s_count = (sprite_id, count)
            # Find color in the character palette, if its not there, add it
            try:
                col_index = SpriteDB.palettes[char].index(color)
            except ValueError:
                SpriteDB.palettes[char].append(color)
                col_index = len(SpriteDB.palettes[char]) - 1
                SpriteDB.color_counts[char].append([])
            # Add the count while keeping list sorted in descending order
            i = 0
            while True:
                if i >= len(SpriteDB.color_counts[char][col_index]):
                    SpriteDB.color_counts[char][col_index].append(s_count)
                    break
                elif SpriteDB.color_counts[char][col_index][i][1] < count:
                    SpriteDB.color_counts[char][col_index].insert(i, s_count)
                    break
                else:
                    i += 1


def load_sprite(filename: str, char: int, desc:  str = '') -> int:
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
    sprite = Sprite(Image.open(filename), char, desc)
    sprite_id = SpriteDB.add_sprite(sprite)
    return(sprite_id)


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
            full_path = os.path.join(directory, filename)
            load_sprite(full_path, char, filename[:-4])


def load_all_sprites(root_path: str = 'images/sprites') -> None:
    """Loads all sprites (naming must be default convention)

    Parameters
    ----------
    root_path: str
        The root folder that contains all sprite folders.
    """
    names = ['Cage', 'Kano', 'Raiden', 'Kang', 'Scorpion', 'Subzero', 'Sonya']
    folders = []
    for x in names:
        folders.append(f'{root_path}/{x}')
    for x in names:
        folders.append(f'{root_path}/{x}/AltSkin')
    folders.append(f'{root_path}/Goro')
    folders.append(f'{root_path}/Tsung')
    for i, folder_path in enumerate(folders):
        load_sprite_folder(folder_path, i)
