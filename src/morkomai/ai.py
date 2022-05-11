import yaml
from mortalkombat import MortalKombat
import time
import random
import string
import recorder

with open('controls.yaml') as f:
    _controls = yaml.safe_load(f)
    CONTROLS = [_controls["player1"], _controls["player2"]]

CHARACTERS = (
    'Johnny Cage',
    'Kano',
    'Raiden',
    'Liu Kang',
    'Scorpion',
    'Sub-Zero',
    'Sonya Blade'
)

AI_STATES = (
    'WAITING_TO_JOIN',
    'SELECT_CHARACTER',
    'INTERMISSION',
    'FIGHTING',
    'EXIT'
)


class AI:
    def __init__(self, game: MortalKombat, player: int,
                 character: int = 5, move_speed: float = 100,
                 reporting: bool = True, captures: bool = True) -> None:
        """Class used to control the AI.

        Parameters
        ----------
        game: MortalKombat
            The application to add the AI to.
        player: int
            The player to control using the AI (Player 1: 0, Player 2: 1).
        character: int, optional
            The character to choose. Defaults to 5 (Sub-Zero).
                0 -> Johnny Cage
                1 -> Kano
                2 -> Raiden
                3 -> Liu Kang
                4 -> Scorpion
                5 -> Sub-Zero
                6 -> Sonya Blade
        move_speed: float, optional
            The time in ms to wait after making a move. Defaults to 100.
        reporting: bool, optional
            If True, the AI outputs its actions to console. Defaults to True.
        captures: bool, optional
            If True, the AI will take screenshots periodically for training
            purposes.

        Attributes
        ----------
        game: MortalKombat
            The application that the AI is playing in.
        player: int
            The player the AI is controlling (Player 1: 0, Player 2: 1).
        move_speed: float
            The time in ms to wait after making a move.
        reporting: bool
            If True, the AI outputs its actions to console.
        captures: bool, optional
            If True, the AI will take screenshots periodically for training
            purposes.
        prev_message: list[str, int]
            prev_message[0]: str
                The previous message text.
            prev_message[1]: int
                The number of times the message has been repeated.
        character: int
            The character that the AI plays.
                0 -> Johnny Cage
                1 -> Kano
                2 -> Raiden
                3 -> Liu Kang
                4 -> Scorpion
                5 -> Sub-Zero
                6 -> Sonya Blade
        state: int
            The current state of the AI
                0 -> WAITING_TO_JOIN
                1 -> SELECT_CHARACTER
                2 -> INTERMISSION
                3 -> FIGHTING
        """
        self.game = game
        self.player = player
        self.move_speed = move_speed
        self.reporting = reporting
        self.captures = captures
        self.prev_message = ['', 0]
        self.character = character
        self.state = 0

    def _get_character(self) -> int: return(self._character)

    def _set_character(self, value: int):
        if type(value) is not int:
            raise TypeError('Character must be an integer in [0, 6]')
        elif not 0 <= value <= 6:
            raise ValueError('Character must be an integer in [0, 6]')
        self._character = value
    character = property(fget=_get_character, fset=_set_character)

    def _get_state(self) -> int: return(self._state)

    def _set_state(self, value: int) -> None:
        max_index = len(AI_STATES) - 1
        err_msg = f'State must be an integer in [0, {max_index}]'
        if type(value) is not int:
            raise TypeError(err_msg)
        elif not 0 <= value <= max_index:
            raise ValueError(err_msg)
        self._state = value
    state = property(fget=_get_state, fset=_set_state)

    def _send_control(self, control: str) -> None:
        """Sends controls to dosbox, keystrokes depend on controls.yaml file.

        Parameters
        ----------
        control: str
            The control to send.
        """
        if self.captures:
            self.random_capture(0.1)
        self.report(control)
        self.game._dosbox.keystroke(CONTROLS[self.player][control])
        time.sleep(self.move_speed / 1000)

    def random_capture(self, p: float, file_name: str = None):
        """Roll a value in [0, 1], if its greater than p, take a screenshot.

        Saves the screenshot to the capture folder set in settings.yaml.

        Parameters
        ----------
        p: float
            The probability of taking a screenshot.
        file_name: str, optional
            The name of the file, defaults to a random string of ascii letters
            and digits starting with an underscore.
        """
        if not 0 <= p <= 1:
            raise ValueError('Invalid probability')
        if random.random() > (1 - p):
            chars = string.ascii_letters + string.digits
            file_name = ''.join(random.choice(chars) for i in range(15))
            self.game.capture(f'{self.game.capture_folder}/_{file_name}.png')

    def report(self, msg: str) -> None:
        """Prints messages to console.

        For messages that are repeated, the message and number of repeats will
        appear on one line to avoid clogging the console.

        Parameters
        ----------
        msg: str
            The message to be printed
        """
        if self.reporting:
            if msg == self.prev_message[0]:
                print('\r' * 500, end='')
                self.prev_message[1] += 1
                print(f'Player {self.player + 1}: {msg} (Repeated '
                      f'x{[self.prev_message[1]]})', end='')
            else:
                print()
                print(f'Player {self.player + 1}: {msg}', end='')
            self.prev_message = [msg, 1]

    # Game controls
    def idle(self) -> None: time.sleep(self.move_speed / 1000)

    def left(self) -> None: self._send_control('left')

    def right(self) -> None: self._send_control('right')

    def crouch(self) -> None: self._send_control('crouch')

    def jump(self) -> None: self._send_control('jump')

    def block(self) -> None: self._send_control('block')

    def high_punch(self) -> None: self._send_control('high_punch')

    def low_punch(self) -> None: self._send_control('low_punch')

    def high_kick(self) -> None: self._send_control('high_kick')

    def low_kick(self) -> None: self._send_control('low_kick')

    # State functions
    def wait_to_join(self) -> None:
        """Wait for the game to start to join the game"""
        join_screen = recorder.open_image(self.game.join_screen)
        char_select = recorder.open_image(self.game.character_screen)
        status = 'Waiting to join game'

        while True:
            time.sleep(1)
            self.report(status)
            current_screen = self.game._recorder.get_current_image()
            at_join_screen = recorder.images_similar(current_screen,
                                                     join_screen, 0.1)
            at_char_screen = recorder.images_similar(current_screen,
                                                     char_select, 0.2)

            if at_join_screen:
                self.join()
                status = 'Waiting for character select screen'
            elif at_char_screen:
                self.join()  # In case 2 AIs are playing
                self.state = 1
                break

    def join(self) -> None:
        self.report('Joining game')
        self.game.join(self.player)

    def character_select(self, character: int) -> None:
        """Navigate to character and select it.

        Parameters
        ----------
        character: int
            The index of the character to select.
                0 -> Johnny Cage
                1 -> Kano
                2 -> Raiden
                3 -> Liu Kang
                4 -> Scorpion
                5 -> Sub-Zero
                6 -> Sonya Blade
        """
        self.report(f'Selecting {CHARACTERS[character]}')
        position = (0, 5)[self.player]  # Where P1/P2 cursors start
        while position != character:
            if position > character:
                if position == 2:
                    self.jump()
                elif position == 5:
                    self.crouch()
                else:
                    self.left()
                position -= 1
            elif position < character:
                if position == 1:
                    self.crouch()
                elif position == 4:
                    self.jump()
                else:
                    self.right()
                position += 1
            time.sleep(0.1)
        self.low_kick()  # Confirm
        self.state = 2

    def wait_to_fight(self) -> None:
        # TODO Lookout for the fight prompt
        self.report('Waiting to start fight')
        time.sleep(3)
        self.report('Fight started')
        self.state = 3

    def fight(self) -> None: raise NotImplementedError

    def run(self) -> None:
        while True:
            if self.state == 0:             # WAITING_TO_JOIN
                self.wait_to_join()
            elif self.state == 1:           # SELECT_CHARACTER
                self.character_select(self.character)
            elif self.state == 2:           # INTERMISSION
                self.wait_to_fight()
            elif self.state == 3:           # FIGHTING
                self.fight()
            elif self.state == 4:           # EXIT
                break
            else:
                raise RuntimeError(f'Invalid state: {self.state}')

    # Fight Tactics
    def random_moves(self) -> None:
        while True:
            choice = random.choice(list(range(18)))
            if choice == 0:
                self.left()
            elif choice == 1:
                self.right()
            elif choice == 2:
                self.jump()
            elif choice == 3:
                self.crouch()
            elif choice == 4:
                self.block()
            elif choice == 5:
                self.high_kick()
            elif choice == 6:
                self.high_punch()
            elif choice == 7:
                self.low_kick()
            elif choice == 8:
                self.low_punch()
            else:
                self.idle()
