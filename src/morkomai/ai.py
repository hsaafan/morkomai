import random
import yaml

from .enumerations import AI_STATES, CHARACTERS

with open('controls.yaml') as f:
    _controls = yaml.safe_load(f)
    CONTROLS = [_controls["player1"], _controls["player2"]]


class AI:
    def __init__(self, controller, player: int,
                 character: int = 5, reporting: bool = True) -> None:
        """Class used to control the AI.

        Parameters
        ----------
        controller: Controller
            The controller to attach the AI to.
        player: int
            The player to control using the AI (Player 1: 0, Player 2: 1).
        character: int, optional
            The character to choose. Default is 5 (SubZero). See CHARACTERS.
        reporting: bool, optional
            If True, the AI outputs its actions to console. Defaults to True.

        Attributes
        ----------
        controller: Controller
            The controller that the AI is attached to.
        player: int
            The player the AI is controlling (Player 1: 0, Player 2: 1).
        reporting: bool
            If True, the AI outputs its actions to console.
        prev_message: list[str, int]
            prev_message[0]: str
                The previous message text.
            prev_message[1]: int
                The number of times the message has been repeated.
        character: int
            The character that the AI plays. See CHARACTERS.
        game_info: list
            List with player 1 info at index 0 and player 2 info at index 2.
            The player information is a list ordered as follows.
                Health: float
                    Player health from 0-100.
                Sprite ID: int
                    The current sprite of the player.
                Bounding Box: tuple
                    The bounding box of the player sprite (x1, y1, x2, y2).
        state: int
            The current state of the AI. See AI_STATES.
        """
        self.controller = controller
        self.player = player
        self.reporting = reporting
        self.prev_message = ['', 0]
        self.character = character
        self.game_info = []
        self.state = AI_STATES['OTHER']

    def _get_character(self) -> int: return(self._character)

    def _set_character(self, value: int) -> None:
        if type(value) is not int:
            raise TypeError('Character must be an integer in [0, 6]')
        elif not 0 <= value <= 6:
            raise ValueError('Character must be an integer in [0, 6]')
        self._character = value
    character = property(fget=_get_character, fset=_set_character,
                         doc="The character that the AI plays.")

    def _get_state(self) -> int: return(self._state)

    def _set_state(self, value: int) -> None:
        max_index = len(AI_STATES) - 1
        err_msg = f'State must be an integer in [0, {max_index}]'
        if type(value) is not int:
            raise TypeError(err_msg)
        elif not 0 <= value <= max_index:
            raise ValueError(err_msg)
        if self.player == 0:
            AI.first_player_state = value
        self._state = value
    state = property(fget=_get_state, fset=_set_state,
                     doc="The current state of the AI.")

    def _send_control(self, control: str) -> None:
        """Sends controls to dosbox, keystrokes depend on controls.yaml file.

        Parameters
        ----------
        control: str
            The control to send.
        """
        self.report(control)
        self.controller.queue_keystroke(CONTROLS[self.player][control])

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
                      f'x{self.prev_message[1]})', end='')
            else:
                print()
                print(f'Player {self.player + 1}: {msg}', end='')
                self.prev_message = [msg, 1]

    # Game controls
    def idle(self) -> None: self._send_control('idle')

    def left(self) -> None: self._send_control('left')

    def right(self) -> None: self._send_control('right')

    def crouch(self) -> None: self._send_control('crouch')

    def jump(self) -> None: self._send_control('jump')

    def block(self) -> None: self._send_control('block')

    def high_punch(self) -> None: self._send_control('high_punch')

    def low_punch(self) -> None: self._send_control('low_punch')

    def high_kick(self) -> None: self._send_control('high_kick')

    def low_kick(self) -> None: self._send_control('low_kick')

    def join(self) -> None: self._send_control('join')

    def character_select(self, character: int) -> None:
        """Navigate to character and select it.

        Parameters
        ----------
        character: int
            The index of the character to select. See CHARACTERS.
        """
        self.report(f'Selecting {CHARACTERS[character]}')
        position = (1, 5)[self.player]  # Where P1/P2 cursors start
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
        self.low_kick()  # Confirm
        self.state = AI_STATES['INTERMISSION']

    def fight(self) -> None: raise NotImplementedError

    def step(self) -> None:
        """Take an action based on AI state."""
        if self.state == AI_STATES['OTHER']:
            pass
        elif self.state == AI_STATES['JOINING']:
            self.join()
        elif self.state == AI_STATES['SELECT_CHARACTER']:
            self.character_select(self.character)
        elif self.state == AI_STATES['INTERMISSION']:
            pass
        elif self.state == AI_STATES['FIGHTING']:
            self.fight()
        else:
            raise RuntimeError(f'Invalid state: {self.state}')

    # Fight Tactics
    def random_moves(self) -> None:
        """AI will pick random controls when playing."""
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

    def update_info(self, info: list) -> None:
        """Update the AI with game state information.

        Parameters
        ----------
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
        self.game_info = info
