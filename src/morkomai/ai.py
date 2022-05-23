import time
import random
import yaml

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

AI_STATES = {
    'OTHER': 0,
    'JOINING': 1,
    'SELECT_CHARACTER': 2,
    'INTERMISSION': 3,
    'FIGHTING': 4,
}


class AI:
    ai_players = 0
    first_player_state = 0

    def __init__(self, controller, player: int,
                 character: int = 5, move_speed: float = 100,
                 reporting: bool = True) -> None:
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
        self.controller = controller
        self.player = player
        self.move_speed = move_speed
        self.reporting = reporting
        self.prev_message = ['', 0]
        self.character = character
        self.state = AI_STATES['OTHER']
        AI.ai_players += 1

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
        if self.player == 0:
            AI.first_player_state = value
        self._state = value
    state = property(fget=_get_state, fset=_set_state)

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

    def join(self) -> None: self._send_control('join')

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

    def update_info(self, info):
        pass
