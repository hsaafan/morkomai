# MORKOMAI - Mortal Kombat AI

MORKOMAI is an AI designed to learn to play the original Mortal Kombat on DOS. The game is not included in this repository, a copy can be bought from GOG.

## Modes

The following modes can be used to train a model or to showcase a trained one.

**Adversarial**: Pits 2 AIs against each other (planned)

**Ladder**: Pits MORKOMAI against the original AI in ladder mode. (planned)

**Human Player**: The AI will take over the role of player 2 while player 1 is controlled by a human opponent. (planned)

## AI Techniques Used

**OCR** is used to detect game messages and keep track of score.

**Transfer Learning** is used to allow the AI to play different characters.

**Computer Vision** is used to detect game state (positions, actions, and characters).

**Reinforcment Learning** is used to train the AI.

**LSTM** is included within the network in an effort to coax the AI into learning combos.

## Setup

1. Clone repository
2. Create virtual environment

    ```bash
    python -m venv ./venv
    source ./venv/bin/activate
    ```

3. Install requirements

    ```bash
    pip install -r requirements.txt
    sudo apt install Xephyr xdotool scrot dosbox
    ```

4. Create directory for game and screenshots

    ```bash
    mkdir dosprogs .captures
    ```

5. Copy game files to folder

    ```bash
    cp /path/to/game/MK1/* dosprogs
