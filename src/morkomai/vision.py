import numpy as np
from math import floor
from PIL import Image
import PIL
import os
import numpy as np
import time


START_POSITIONS = (         # Where players start each round
    (190, 220, 280, 340),
    (350, 220, 440, 340)
)


def sample_pixels(array: np.ndarray, ppr: int, ppc: int,
                  x_offset: int = 0, y_offset: int = 0) -> list:
    """Return a uniform set of values in an array.

    Parameters
    ----------
    array: np.ndarray
        The array to sample from.
    ppr: int
        Number of rows to take samples from.
    ppc: int
        Number of columns to take samples from.
    x_offset: int, optional
        Adds an offset to x coordinates in returned list.
    y_offset: int, optional
        Adds an offset to y coordinates in returned list.

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


class Sprite:
    def __init__(self, img: Image.Image, char: int, desc: str = '') -> None:
        """Class for sprite images.

        Parameters
        ----------
        img: PIL.Image.Image
            The sprite image.
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
        mask: np.ndarray
            A mask that is True for all non-blank pixels and False otherwise.
            A blank pixel is one which has a value of 0 in all channels.
        non_blanks: int
            The number of non blank pixels in the sprite.
        colors: zip
            A zipped iterable with the first value containing color arrays and
            the second containing a count of how many times the color appears
            in the sprite, and the third value containing the percentage of
            the sprite (excluding blank pixels) that the color covers.
        """
        w, h = img.size
        c = len(img.mode)
        self.data = np.asarray(img.getdata()).reshape((h, w, c))
        self.char = char
        self.desc = desc
        mask = np.ones((h, w), bool)
        blanks = np.all(self.data == 0, axis=-1)
        mask[blanks] = False
        self.mask = mask
        self.non_blanks = np.sum(self.mask)
        self.extract_palette()

    def extract_palette(self) -> None:
        """Store the colors and percent of sprite they take up."""
        colors, counts = np.unique(self.data.reshape((-1, 3)),
                                   return_counts=True, axis=0)
        p_color = counts / self.non_blanks
        self.colors = zip(colors, counts, p_color)


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
    sprites = [[] for _ in range(16)]
    palettes = [[] for _ in range(16)]
    color_counts = [[] for _ in range(16)]
    p_color = [[] for _ in range(16)]

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
        """Adds the sprite colors to the palettes, color_counts and p_color
        attributes.

        Parameters
        ----------
        sprite: Sprite
            The sprite object to extract the colors from.
        sprite_id: int
            The id of the sprite to extract the colors from.
        """
        char = sprite.char
        for color, count, p_color in sprite.colors:
            s_count = (sprite_id, p_color)
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
    def __init__(self, p1_char: int, p2_char: int,
                 ppr: int = 3, ppc: int = 3, timeout: float = 0.15) -> None:
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
        timeout: float, optional
            Seconds to wait per player before halting search, defaults to 0.15.

        Attributes
        ----------
        img: np.ndarray
            The image to match sprites to. Shape is (height, width, channels).
        timeout: float
            Seconds to wait per player before halting search.
        """
        self._ppr = ppr
        self._ppc = ppc
        self.timeout = timeout
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
            pixels = sample_pixels(bounded_img, self._ppr, self._ppc, l, t)

            if only_p1:
                bounding_boxes = [self.match(pixels, self._p1_char)]
            elif only_p2:
                bounding_boxes = [self.match(pixels, self._p2_char)]
        return(bounding_boxes)

    def match(self, pixels: list, char: int, color_tol: int = 5) -> tuple:
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
        start = time.time()
        pixel_order = []
        for x, y, color in pixels:
            # Order sampled sprites by how rare they are in the character
            # sprites
            if np.all(color == 0):
                continue  # Dont match empty spots (black in RGB)
            try:
                c = self.find_color_matches(SpriteDB.palettes[char],
                                            color, color_tol)[0][0]
            except IndexError:
                continue  # Color doesn't match any in character palette
            count = SpriteDB.color_counts[char][c]
            new_pixel = (x, y, c)
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

        for x, y, c in pixel_order:
            sprite_p_color = SpriteDB.p_color[char][c]
            for i, _ in sprite_p_color:
                # Search in descending order of most color coverage in sprite
                sprite = SpriteDB.sprites[char][i]
                sprite_match = self.find_img((x, y), sprite)
                if sprite_match is not None:
                    return(i, sprite_match)
                if time.time() - start > self.timeout:
                    break
        return(None)

    def find_img(self, coords: tuple, sprite: Sprite,
                 min_match: float = 0.5, color_tol: int = 5) -> tuple:
        """Attempt to match a sprite to a location on an image.

        The match starts at the given coordinates on the image and iterates
        through the pixels on the sprite that match the color of the pixel. The
        match box grows with each iteration until either the min_match is not
        met, or the full sprite is matched.

        Parameters
        ----------
        coords: tuple
            The image pixel coordinates to match to. Values are (x, y).
        sprite: Sprite
            Sprite to attempt match on.
        min_match: float, optional
            The percent of pixels that need to be matched for the sprite to
            match to the image. Defaults to 0.5. Setting this too low will
            increase search time but setting it to high will not match when
            characters are off screen or covered.
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
        sprite_h, sprite_w, sprite_channels = sprite.data.shape
        img = self.img
        _, _, channels = img.shape
        if sprite_channels != channels:
            raise TypeError('Number of channels do not match')

        x, y = coords
        match_coords = self.find_color_matches(sprite.data, img[y, x],
                                               color_tol)
        sprite_x1, sprite_y1, sprite_x2, sprite_y2 = (0, 0, 0, 0)
        crop_x1, crop_y1, crop_x2, crop_y2 = (0, 0, 0, 0)

        for match_y, match_x in zip(match_coords[0], match_coords[1]):
            sprite_mask = sprite.mask.copy()  # Blanks in sprite
            rad = 1
            prev_tiles = 0
            prev_matches = 0
            # Maximum radius to cover image
            target_radius = np.max([sprite_w - match_x, match_x,
                                    sprite_h - match_y, match_y])
            # Crop the image to what the sprite bounding box would be
            crop_x1 = x - match_x
            crop_y1 = y - match_y
            crop_x2 = crop_x1 + sprite_w
            crop_y2 = crop_y1 + sprite_h
            cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            psprite_x1 = None  # No previous check applied on sprite

            while True:
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
                sprite_view = sprite.data[sprite_y1:sprite_y2,
                                          sprite_x1:sprite_x2]
                image_view = cropped_img[sprite_y1:sprite_y2,
                                         sprite_x1:sprite_x2]
                mask = sprite_mask[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
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
                n_tiles = np.sum(mask) + prev_tiles
                # Compare
                diff = np.abs(sprite_view - image_view)
                new_matches = np.sum(np.max(diff, axis=-1) < color_tol)
                matches = prev_matches + new_matches
                try:
                    match_pcnt = matches / n_tiles
                except ZeroDivisionError:
                    # This technicaly should never happen since the first pixel
                    # will always match
                    match_pcnt = 1
                if match_pcnt < min_match:
                    break  # Sprite doesn't match here, go to next coordinate
                elif rad >= target_radius:
                    # The whole image has been checked and matches
                    return(x - match_x, y - match_y,
                           x - match_x + sprite_w, y - match_y + sprite_h)
                # Increase radius and store previous check values
                rad *= 2
                psprite_x1 = sprite_x1
                psprite_y1 = sprite_y1
                psprite_x2 = sprite_x2
                psprite_y2 = sprite_y2
                prev_matches = matches
                prev_tiles = n_tiles
        return(None)  # No matches on sprite

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
        coords: tuple
            A tuple of indices where matches occur. The size of the tuple is
            equal to 1 less than the number of dimensions of the image.
        """
        above = np.all(image >= color - color_tol, axis=-1)
        below = np.all(image <= color + color_tol, axis=-1)
        coords = np.nonzero(above * below)
        return(coords)

    def update_image(self, img: Image) -> None:
        """Update the stored image.

        Parameters
        ----------
        img: PIL.Image.Image
            The image to store.
        """
        width, height = img.size
        self.img = np.asarray(img.getdata()).reshape((height, width, 3))


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
    flipped_img = sprite_img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
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
