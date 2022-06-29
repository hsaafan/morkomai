import random
import torch
from collections import deque

from .globals import *
from . import vision


class AI:
    """Class used to control the AI.

    Attributes
    ----------
    controller: Controller
        The controller that the AI is attached to.
    lstm_model: MorkomAI
        The LSTM model to use for choosing actions.
    player: int
        The player the AI is controlling (Player 1: 0, Player 2: 1).
    reporting: bool
        If True, the AI outputs its actions to console.
    prev_message: list[str, int]
        prev_message[0]: str
            The previous message text.
        prev_message[1]: int
            The number of times the message has been repeated.
    character: str
        The character that the AI plays. See CHARACTERS.
    enemy_character: str
        The character that the opponent plays. See CHARACTERS.
    game_info: list
        List with player 1 info at index 0 and player 2 info at index 2. The
        player information is a list ordered as follows.
            Health: float
                Player health from 0-1.
            Sprite ID: int
                The current sprite of the player.
            Bounding Box: tuple
                The bounding box of the player sprite (x1, y1, x2, y2).
    prev_info: list
        The previous game_info array.
    toggled_controls: dict
        A dictionary with boolean variables indicating whether keys are
        currently held down.
    state: int
        The current state of the AI. See AI_STATES.
    match_memory: deque
        Contains all the states of the current match.
    prev_input: int
        The previous input, the integer value corresponds to the ai fight
        function choice.
    prev_reward: float
        The previous reward value.
    prev_state: torch.tensor
        The previous match state.
    lstm_info_state: tuple
        The current state for lstm_model.net.lstm_info.
    lstm_controls_state: tuple
        The current state for lstm_model.net.lstm_controls.
    prev_lstm_info_state: tuple
        The previous state for lstm_model.net.lstm_info.
    prev_lstm_controls_state: tuple
        The previous state for lstm_model.net.lstm_controls.


    Parameters
    ----------
    controller: Controller
        The controller to attach the AI to.
    player: int
        The player to control using the AI (Player 1: 0, Player 2: 1).
    character: str, optional
        The character to choose. Default is Johnny Cage. See CHARACTERS.
    reporting: bool, optional
        If True, the AI outputs its actions to console. Defaults to True.
    """
    def __init__(self, controller, player: int, character: str = 'Johnny Cage',
                 reporting: bool = True) -> None:
        self.controller = controller
        self.lstm_model = self.controller.lstm_model
        self.player = player
        self.reporting = reporting
        self.prev_message = ['', 0]
        self.character = character
        self.enemy_character = 'Johnny Cage'
        self.game_info = []
        self.prev_info = []
        self.toggled_controls = {c: False for c in CONTROLS[player].keys()}
        self.state = AI_STATES['OTHER']
        self.reset_lstm_state()
        self.match_memory = deque()

    def _get_character(self) -> str: return(self._character)

    def _set_character(self, value: str) -> None:
        self._character = value
        self.lstm_model.character = CHARACTER_IDS[value]
    character = property(fget=_get_character, fset=_set_character,
                         doc="The character that the AI plays.")

    def _get_enemy_character(self) -> str: return(self._enemy_character)

    def _set_enemy_character(self, value: str) -> None:
        self._enemy_character = value
    enemy_character = property(fget=_get_enemy_character,
                               fset=_set_enemy_character,
                               doc="The character that the enemy plays.")

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

    def _send_toggle_control(self, control: str) -> None:
        """Sends controls to dosbox, keystrokes depend on controls.yaml file.

        Parameters
        ----------
        control: str
            The control to send.
        """
        new_state = not self.toggled_controls[control]
        self.report(f'Toggling {control} {"down" if new_state else "up"}')
        self.controller.queue_toggle_keystroke((CONTROLS[self.player][control],
                                               new_state))
        self.toggled_controls[control] = new_state

    def reset_lstm_state(self) -> None:
        """Resets the lstm state of the AI."""
        states = self.lstm_model.get_random_states()
        p_states = self.lstm_model.get_random_states()
        self.lstm_info_state, self.lstm_controls_state = states
        self.prev_lstm_info_state, self.prev_lstm_controls_state = p_states
        self.prev_input = 0
        self.prev_reward = 0
        self.prev_state = torch.tensor([START_POSITIONS[0][0] / ARENA_SIZE[0],
                                        START_POSITIONS[0][1] / ARENA_SIZE[1],
                                        65 / ARENA_SIZE[0],
                                        100 / ARENA_SIZE[1],
                                        1,  # Character health
                                        START_POSITIONS[1][0] / ARENA_SIZE[0],
                                        START_POSITIONS[1][1] / ARENA_SIZE[1],
                                        65 / ARENA_SIZE[0],
                                        100 / ARENA_SIZE[1],
                                        1,  # Enemy health
                                        0, 0, 0, 0,  # Toggle status
                                        0,  # Character Sprite ID
                                        self.prev_input,
                                        CHARACTER_IDS[self.character],
                                        0,  # Enemy Sprite ID
                                        CHARACTER_IDS[self.enemy_character]]
                                       ).unsqueeze(0).unsqueeze(0).float()

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

    # Toggle controls
    def toggle_left(self) -> None:
        for control in ['block', 'right', 'crouch']:
            if self.toggled_controls[control]:
                self._send_toggle_control(control)
        self._send_toggle_control('left')

    def toggle_right(self) -> None:
        for control in ['block', 'left', 'crouch']:
            if self.toggled_controls[control]:
                self._send_toggle_control(control)
        self._send_toggle_control('right')

    def toggle_block(self) -> None:
        for control in ['left', 'right']:
            if self.toggled_controls[control]:
                self._send_toggle_control(control)
        self._send_toggle_control('block')

    def toggle_crouch(self) -> None:
        for control in ['left', 'right']:
            if self.toggled_controls[control]:
                self._send_toggle_control(control)
        self._send_toggle_control('crouch')

    def character_select(self, character: str) -> None:
        """Navigate to character and select it.

        Parameters
        ----------
        character: str
            The character to select. See CHARACTERS.
        """
        self.report(f'Selecting {character}')
        img = vision.convert_to_array(self.controller.img)
        position = vision.detect_character_select_positions(img)[self.player]
        if not (0 <= CHARACTER_IDS[character] <= 6):
            raise RuntimeError(f'Cannot select character: {character}')
        if position is None:
            return
        for _ in range(2):
            # To prevent moving cursor too early
            self.idle()
        while position != CHARACTER_IDS[character]:
            if position > CHARACTER_IDS[character]:
                if position == 2:
                    self.jump()
                elif position == 5:
                    self.crouch()
                else:
                    self.left()
                position -= 1
            elif position < CHARACTER_IDS[character]:
                if position == 1:
                    self.crouch()
                elif position == 4:
                    self.jump()
                else:
                    self.right()
                position += 1
        self.low_kick()  # Confirm
        self.state = AI_STATES['INTERMISSION']

    def fight(self) -> None:
        """Function is overwritten by the AI tactics set."""
        raise NotImplementedError

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
        choice = random.choice(list(range(14)))
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
        elif choice == 9:
            self.block()
        elif choice == 10:
            self.toggle_block()
        elif choice == 11:
            self.toggle_left()
        elif choice == 12:
            self.toggle_right()
        elif choice == 13:
            self.toggle_crouch()
        else:
            self.idle()
        self.prev_input = choice

    def lstm_ai(self, update_model: bool = True) -> None:
        """AI will pick move according to LSTM model."""
        char_health, char_spid, char_bbox = self.game_info[self.player]
        enemy_health, enemy_spid, enemy_bbox = self.game_info[not self.player]

        # Contruct game state tensor
        state = torch.tensor([char_bbox[0] / ARENA_SIZE[0],
                              char_bbox[1] / ARENA_SIZE[1],
                              char_bbox[2] - char_bbox[0] / ARENA_SIZE[0],
                              char_bbox[3] - char_bbox[1] / ARENA_SIZE[1],
                              char_health,
                              enemy_bbox[0] / ARENA_SIZE[0],
                              enemy_bbox[1] / ARENA_SIZE[1],
                              enemy_bbox[2] - enemy_bbox[0] / ARENA_SIZE[0],
                              enemy_bbox[3] - enemy_bbox[1] / ARENA_SIZE[1],
                              enemy_health,
                              self.toggled_controls['block'],
                              self.toggled_controls['left'],
                              self.toggled_controls['right'],
                              self.toggled_controls['crouch'],
                              char_spid,
                              self.prev_input,
                              CHARACTER_IDS[self.character],
                              enemy_spid,
                              CHARACTER_IDS[self.enemy_character]
                              ]).unsqueeze(0).unsqueeze(0).float()
        net_returns = self.lstm_model.act(state, self.lstm_info_state,
                                          self.lstm_controls_state)
        self.prev_lstm_info_state = self.lstm_info_state
        self.prev_lstm_controls_state = self.lstm_controls_state
        choice, lstm_info_state, lstm_controls_state = net_returns
        self.lstm_info_state = lstm_info_state
        self.lstm_controls_state = lstm_controls_state

        if update_model:
            self.update_ai_memory(state)
            _, loss = self.lstm_model.learn()
            if loss is not None:
                self.report(f'Loss = {loss}')

        if choice == 0:
            self.idle()
        elif choice == 1:
            self.left()
        elif choice == 2:
            self.right()
        elif choice == 3:
            self.crouch()
        elif choice == 4:
            self.jump()
        elif choice == 5:
            self.low_kick()
        elif choice == 6:
            self.high_kick()
        elif choice == 7:
            self.low_punch()
        elif choice == 8:
            self.high_punch()
        elif choice == 9:
            self.block()
        elif choice == 10:
            self.toggle_block()
        elif choice == 11:
            self.toggle_left()
        elif choice == 12:
            self.toggle_right()
        elif choice == 13:
            self.toggle_crouch()
        else:
            raise RuntimeError(f'Received invalid control: {choice}')
        self.prev_input = choice

        if update_model:
            prev_char_health = self.game_info[self.player][0]
            prev_enemy_health = self.game_info[not self.player][0]
            self.prev_reward = (5 * (prev_enemy_health - enemy_health)
                                - (prev_char_health - char_health))
            self.prev_state = state

    def update_ai_memory(self, state: torch.tensor) -> None:
        """Adds the current state transition to internal memory.

        Parameters
        ----------
        state: torch.tensor
            The current match state.
        """
        p_state = torch.clone(self.prev_state)
        p_info_state = (torch.clone(x) for x in self.prev_lstm_info_state)
        p_controls_state = (torch.clone(x) for x in
                            self.prev_lstm_controls_state)
        state = torch.clone(state)
        info_state = (torch.clone(x) for x in self.lstm_info_state)
        controls_state = (torch.clone(x) for x in self.lstm_controls_state)

        self.match_memory.append((p_state, p_info_state, p_controls_state,
                                  state, info_state, controls_state,
                                  self.prev_input, self.prev_reward))

    def update_lstm_memory(self) -> None:
        """Dumps the match memory to the lstm model."""
        while self.match_memory:
            self.lstm_model.cache(*self.match_memory.popleft())

    def update_info(self, info: list) -> None:
        """Update the AI with game state information.

        Parameters
        ----------
        info: list
            List with player 1 info at index 0 and player 2 info at index 2.
            The player information is a list ordered as follows.
                Health: float
                    Player health from 0-1.
                Sprite ID: int
                    The current sprite of the player.
                Bounding Box: tuple
                    The bounding box of the player sprite (x1, y1, x2, y2).
        """
        self.prev_info = self.game_info
        self.game_info = info

    def reset_info(self) -> None:
        """Reset the game info lists"""
        self.prev_info = [[1, 0, START_POSITIONS[0]],
                          [1, 0, START_POSITIONS[1]]]
        self.game_info = [[1, 0, START_POSITIONS[0]],
                          [1, 0, START_POSITIONS[1]]]
