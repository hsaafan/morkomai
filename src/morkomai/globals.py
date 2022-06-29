import yaml

AI_STATES = {
    'OTHER': 0,
    'JOINING': 1,
    'SELECT_CHARACTER': 2,
    'INTERMISSION': 3,
    'FIGHTING': 4,
}

ARENA_SIZE = (320, 200)

CHARACTERS = {
    0: 'Johnny Cage',
    1: 'Kano',
    2: 'Raiden',
    3: 'Liu Kang',
    4: 'Scorpion',
    5: 'Sub-Zero',
    6: 'Sonya Blade',
    7: 'Goro',
    8: 'Shang Tsung',
    # Alternates are just used for sprites
    9: 'Johnny Cage (Alternate)',
    10: 'Kano (Alternate)',
    11: 'Raiden (Alternate)',
    12: 'Liu Kang (Alternate)',
    13: 'Scorpion (Alternate)',
    14: 'Sub-Zero (Alternate)',
    15: 'Sonya Blade (Alternate)',
}

CHARACTER_IDS = {name: id for id, name in CHARACTERS.items()}

CHARACTER_SELECT_FRAMES = {
    'Johnny Cage': (186, 174, 239, 237),
    'Kano': (240, 174, 293, 237),
    'Raiden': (240, 238, 293, 301),
    'Liu Kang': (294, 238, 347, 301),
    'Scorpion': (347, 238, 400, 301),
    'Sub-Zero': (347, 174, 400, 237),
    'Sonya Blade': (401, 174, 454, 237)
}

CONTINUE_POSITION = (251, 192, 390, 278)

with open('controls.yaml') as f:
    _controls = yaml.safe_load(f)
    CONTROLS = [_controls["player1"], _controls["player2"]]

DB_PICKLE_FILENAME = 'SpriteDB.pickle'

HEALTH_BAR_POSITIONS = (
    (181, 163, 309, 171),
    (331, 163, 459, 171)
)

PLAY_AREA = (160, 170, 480, 330)

SCENES = {
    'OTHER': 0,
    'INTRODUCTION': 1,
    'CHARACTER_SELECT': 2,
    'FIGHT_PROMPT': 3,
    'GAME_OVER': 4
}

SPRITE_DESC = {
    0: "idle",
    # Misc Actions
    1: "crouch",
    2: "diagjump",
    3: "jump",
    4: "walk",

    # Blocking
    5: "block",
    6: "crouchblock",

    # Damaged
    7: "bodied",
    8: "ballbroken",
    9: "crouchdamaged",
    10: "glassjaw",
    11: "jawed",
    12: "sweeped",
    13: "thrown",

    # Standard Attacks
    14: "elbow",
    15: "flykick",
    16: "flypunch",
    17: "highkick",
    18: "highpunch",
    19: "jumpkick",
    20: "kickstart",
    21: "knee",
    22: "lowkick",
    23: "lowpunch",
    24: "punchstart",
    25: "roundhouse",
    26: "sweep",
    27: "throw",
    28: "tiger",
    29: "uppercut",

    # Cage
    30: "ballbreaker",
    31: "greenbolt",
    32: "shadowkick",

    # Goro
    33: "gorofireball",
    34: "stomp",

    # Kang
    35: "fireball",

    # Kano
    36: "knifethrow",

    # Raiden
    37: "lightning",
    38: "torpedo",

    # Scorpion
    39: "harpoon",

    # Sonya
    40: "leggrab",
    41: "ringtoss",

    # Sub Zero
    42: "icethrow",
    43: "slide",

    # Shang Tsung
    44: "skulls",
}

SPRITE_IDS = {desc: id for id, desc in SPRITE_DESC.items()}

START_POSITIONS = (         # Where players start each round
    (190, 220, 280, 340),
    (350, 220, 440, 340)
)

TIMER_POSITION = (309, 145, 333, 159)
