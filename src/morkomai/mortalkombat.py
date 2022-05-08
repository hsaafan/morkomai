from dosbox import DOSBox
import time
import random
from typing import Tuple


class MortalKombat:
    def __init__(self, dosbox: DOSBox = None) -> None:
        """Class used to run Mortal Kombat application.

        Parameters
        ----------
        dosbox: DOSBox, optional
            The DOSBox object to run Mortal Kombat on. If none are passed, a
            new DOSBox object (along with a new Display object) will be
            created.

        Attributes
        ----------
        player1: Fighter
            Object that controls player 1.
        player2: Fighter
            Object that controls player 2.
        """
        if dosbox is None:
            dosbox = DOSBox()
        self._dosbox = dosbox
        self.player1 = None
        self.player2 = None

    def start(self, conf_file: str = 'dos.conf') -> None:
        """Starts Mortal Kombat.

        Parameters
        ----------
        conf_file: str, optional
            The path of the dosbox conf file that should include startup
            commands for game. Default conf file included will be run if no
            other are passed.
        """
        if self._dosbox.is_running:
            raise RuntimeError('dosbox is already running')
        self._dosbox.start(conf_file=conf_file)
        # HACK first keystroke seems to get ignored
        self._dosbox.keystroke('Return')

    def wait_for_intro(self) -> None:
        """Wait for the game intro to finish."""
        time.sleep(5)  # TODO: A better way to do this

    def player_start(self, player1: bool = False,
                     player2: bool = True, deadtime: float = 1) -> None:
        """Start players that are controlled by this program.

        Parameters
        ----------
        player1: bool, optional
            Start and control player 1. Defaults to False.
        player2: bool, optional
            Start and control player 2. Defaults to True.
        deadtime: float, optional
            Time (in seconds) to wait before starting second player. Defaults
            to 1.
        """
        if player1:
            self._dosbox.keystroke('F1')
            self.player1 = Fighter(self, 1)
        time.sleep(deadtime)
        if player2:
            self._dosbox.keystroke('F2')
            self.player2 = Fighter(self, 2)

    def stop(self) -> None:
        """Stop Mortal Kombat and the dosbox application"""
        self._dosbox.stop()


class Fighter:
    def __init__(self, game: MortalKombat, player: int) -> None:
        self.game = game
        self.player = player

    def keystroke(self, key1: str, key2: str) -> None:
        if self.player == 1:
            self.game._dosbox.keystroke(key1)
        elif self.player == 2:
            self.game._dosbox.keystroke(key2)

    # Game controls
    def left(self): self.keystroke('z', 'KP_Left')

    def right(self): self.keystroke('c', 'KP_Right')

    def crouch(self): self.keystroke('x', 'KP_Down')

    def jump(self): self.keystroke('s', 'KP_Up')

    def block(self): self.keystroke('j', 'KP_5')

    def high_punch(self): self.keystroke('u', 'KP_7')

    def low_punch(self): self.keystroke('n', 'KP_1')

    def high_kick(self): self.keystroke('i', 'KP_9')

    def low_kick(self): self.keystroke('m', 'KP_3')

    def confirm_character_select(self) -> None:
        """Confirm the character selection."""
        self.keystroke('j', 'KP_5')


def mortal_kombat_startup() -> Tuple[MortalKombat, Fighter, Fighter]:
    mk1 = MortalKombat()
    mk1.start()
    mk1.wait_for_intro()
    print('Starting player(s)')
    mk1.player_start(player1=False, player2=True)
    time.sleep(1)
    print('Confirming characters')
    # mk1.player1.confirm_character_select()
    mk1.player2.confirm_character_select()
    mk1.wait_for_intro()
    return(mk1, mk1.player1, mk1.player2)


if __name__ == "__main__":
    mk1, p1, p2 = mortal_kombat_startup()
    turn = 0
    while True:
        if turn:
            player = p1
        else:
            player = p2
        turn = not turn
        if player is None:
            continue
        choice = random.choice(list(range(9)))
        if choice == 0:
            player.left()
        if choice == 1:
            player.right()
        if choice == 2:
            player.jump()
        if choice == 3:
            player.crouch()
        if choice == 4:
            player.block()
        if choice == 5:
            player.high_kick()
        if choice == 6:
            player.high_punch()
        if choice == 7:
            player.low_kick()
        if choice == 8:
            player.low_punch()
