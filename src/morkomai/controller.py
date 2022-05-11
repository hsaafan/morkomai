from mortalkombat import MortalKombat
import threading
from ai import AI


def start_game(AI_1: bool = False, AI_2: bool = True,
               AI_1_tactics: str = 'random',
               AI_2_tactics: str = 'random') -> MortalKombat:
    threads = []
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

    try:
        for player_thread in threads:
            player_thread.start()
    except ChildProcessError:
        print('\ndosbox or the display server have stopped')
