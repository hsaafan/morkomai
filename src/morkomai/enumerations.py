AI_STATES = {
    'OTHER': 0,
    'JOINING': 1,
    'SELECT_CHARACTER': 2,
    'INTERMISSION': 3,
    'FIGHTING': 4,
}

CHARACTERS = (
    'Johnny Cage',
    'Kano',
    'Raiden',
    'Liu Kang',
    'Scorpion',
    'Sub-Zero',
    'Sonya Blade'
)

HEALTH_BAR_POSITIONS = (
    (181, 163, 309, 171),
    (331, 163, 459, 171)
)

SCENES = {
    'OTHER': 0,
    'INTRODUCTION': 1,
    'CHARACTER_SELECT': 2,
    'FIGHT_PROMPT': 3,
    'GAME_OVER': 4
}

START_POSITIONS = (         # Where players start each round
    (190, 220, 280, 340),
    (350, 220, 440, 340)
)
