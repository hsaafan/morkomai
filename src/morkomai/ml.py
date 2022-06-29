import copy
import random
import glob
import os

import torch
import torch.nn as nn
import numpy as np

from .globals import *


class MKNet(nn.Module):
    """LSTM network class for MorkomAI

    Inherits from torch.nn.Module class

    Attributes
    ----------
    embd_actions_opt: dict
        Contains structure of embd_actions layer.
    embd_chars_opt: dict
        Contains structure of embd_chars layer.
    embd_inputs_opt: dict
        Contains structure of embd_inputs layer.
    lstm_info_opt: dict
        Contains structure of lstm_info layer.
    lstm_controls_opt: dict
        Contains structure of lstm_controls layer.
    fc_net_opt: list
        List of dicts that contain structure of fc_net layers.
    embd_actions: torch.nn.Embedding
        Embedding layer for sprite ids.
    embd_chars: torch.nn.Embedding
        Embedding layer for character ids.
    embd_inputs: torch.nn.Embedding
        Embedding layer for controls.
    lstm_info: torch.nn.LSTM
        LSTM layer for general gamesense.
    lstm_controls: torch.nn.LSTM
        LSTM layer for character knowledge.
    fc_net: torch.nn.Sequential
        Fully connected layers with ReLU and Softmax output for determining
        next player action.
    embd_actions_online: torch.nn.Embedding
        Copy of embd_actions for online use.
    embd_chars_online: torch.nn.Embedding
        Copy of embd_chars for online use.
    embd_inputs_online: torch.nn.Embedding
        Copy of embd_inputs for online use.
    lstm_info_online: torch.nn.LSTM
        Copy of lstm_info for online use.
    lstm_controls_online: torch.nn.LSTM
        Copy of lstm_controls for online use.
    fc_net_online: torch.nn.Sequential
        Copy of fc_net for online use.
    """
    def __init__(self) -> None:
        super().__init__()

        # Embedding layers
        """Embedding layers take in classes and output float tensors."""
        self.embd_actions_opt = {
            'num_embeddings': len(SPRITE_DESC),
            'embedding_dim': 5
        }
        """7 playable characters + 2 bosses = 9 embeddings"""
        self.embd_chars_opt = {
            'num_embeddings':  9,
            'embedding_dim': 5
        }
        """Possible Controls (* indicates toggle)
            idle, <, >, v, ∧, lk, hk, lp, hp, block, block*, <*, >*, v*
        """
        self.embd_inputs_opt = {
            'num_embeddings':  14,
            'embedding_dim': 5
        }
        sz_actions = self.embd_actions_opt['embedding_dim']
        sz_char_id = self.embd_chars_opt['embedding_dim']
        sz_inputs = self.embd_inputs_opt['embedding_dim']

        """First LSTM agnostic of character, should learn general gamesense.
        Inputs:
            Player Info: x, y, w, h, health
            Enemey Info: x, y, w, h, health
            Embeddings:
                Player: action_embd
                Enemey: action_embd, char_id_embd
            State Info: prev_lstm_info_state
        Outputs:
            Layer Output: lstm_info_output
            State Info: lstm_info_state
        """
        self.lstm_info_opt = {
            'input_size': 5 + 5 + sz_actions + sz_actions + sz_char_id,
            'hidden_size': 32,
            'num_layers': 3,
            'batch_first': True
        }
        """Second LSTM combines learned gamesense with character knowledge.
        Inputs:
            Previous Layers: lstm_info_output
            Control Toggle Status: block*, <*, >*, v*
            Embeddings:
                Previous Input: input_embd
                Player: char_id
            State Info: prev_lstm_controls_state
        Outputs:
            Layer Output: lstm_controls_output
            State Info: lstm_controls_state
        """
        self.lstm_controls_opt = {
            'input_size': (self.lstm_info_opt['hidden_size'] + 4
                           + sz_inputs + sz_char_id),
            'hidden_size': 32,
            'num_layers': 3,
            'batch_first': True
        }

        """Final fully connected layers determine actions to take.
        Inputs:
            Previous Layers: lstm_info_output, lstm_controls_output
            Outputs: p_actions
        """
        self.fc_net_opt = [
            {'in_features': (self.lstm_info_opt['hidden_size'] +
                             self.lstm_controls_opt['hidden_size']),
             'out_features': 32},
            {'in_features': 32,
             'out_features': 32},
            {'in_features': 32,
             'out_features': self.embd_inputs_opt['num_embeddings']}
        ]

        # Create offline network
        self.embd_actions = nn.Embedding(**self.embd_actions_opt)
        self.embd_chars = nn.Embedding(**self.embd_chars_opt)
        self.embd_inputs = nn.Embedding(**self.embd_inputs_opt)
        self.lstm_info = nn.LSTM(**self.lstm_info_opt)
        self.lstm_controls = nn.LSTM(**self.lstm_controls_opt)
        self.fc_net = nn.Sequential(nn.Linear(**(self.fc_net_opt[0])),
                                    nn.ReLU(),
                                    nn.Linear(**(self.fc_net_opt[1])),
                                    nn.ReLU(),
                                    nn.Linear(**(self.fc_net_opt[2])),
                                    nn.Softmax(dim=-1))

        # Create online network
        self.embd_actions_online = copy.deepcopy(self.embd_actions)
        self.embd_chars_online = copy.deepcopy(self.embd_chars)
        self.embd_inputs_online = copy.deepcopy(self.embd_inputs)
        self.lstm_info_online = copy.deepcopy(self.lstm_info)
        self.lstm_controls_online = copy.deepcopy(self.lstm_controls)
        self.fc_net_online = copy.deepcopy(self.fc_net)

        # Remove gradient calculations for offline network
        for parameters in [self.embd_actions.parameters(),
                           self.embd_chars.parameters(),
                           self.embd_inputs.parameters(),
                           self.lstm_info.parameters(),
                           self.lstm_controls.parameters(),
                           self.fc_net.parameters()]:
            for p in parameters:
                p.requires_grad = False

    def forward(self, state: torch.tensor,
                prev_lstm_info_state: tuple, prev_lstm_controls_state: tuple,
                online: bool = False) -> tuple:
        """Step the model forward

        Parameters
        ----------
        state: torch.tensor
            Tensor of game state containing:
                Player Info: x, y, w, h, health
                Enemy Info: x, y, w, h, health
                Control Toggle Status: block*, <*, >*, v*
                Player Classes: sprite_id, prev_input, char_id
                Enemy Classes: sprite_id, char_id
        prev_lstm_info_state: tuple
            Previous lstm_info state.
        prev_lstm_controls_state: tuple
            Previous lstm_controls state.
        online: bool, optional
            Perform a step forward using the online network. Defaults to False.

        Returns
        -------
        p_actions: torch.tensor
            Tensor containing probabilities of taking each action.
        lstm_info_state: tuple
            Output state of lstm_info.
        lstm_controls_state: tuple
            Output state of lstm_controls.
        """
        player_sprite = state[:, :, 14].long()
        prev_input = state[:, :, 15].long()
        player_char = state[:, :, 16].long()
        enemy_sprite = state[:, :, 17].long()
        enemy_char = state[:, :, 18].long()
        if online:
            i_info = torch.cat((state[:, :, :10],
                                self.embd_actions_online(player_sprite),
                                self.embd_actions_online(enemy_sprite),
                                self.embd_chars_online(enemy_char)),
                               dim=-1).float()
            out = self.lstm_info_online(i_info, prev_lstm_info_state)
            lstm_info_output, lstm_info_state = out

            i_controls = torch.cat((lstm_info_output,
                                    state[:, :, 10:14],
                                    self.embd_inputs_online(prev_input),
                                    self.embd_chars_online(player_char)),
                                   dim=-1).float()
            out = self.lstm_controls_online(i_controls,
                                            prev_lstm_controls_state)
            lstm_controls_output, lstm_controls_state = out

            lstm_outputs = torch.cat((lstm_info_output,
                                      lstm_controls_output),
                                     dim=-1).float()
            p_actions = self.fc_net_online(lstm_outputs)
        else:
            i_info = torch.cat((state[:, :, :10],
                                self.embd_actions(player_sprite),
                                self.embd_actions(enemy_sprite),
                                self.embd_chars(enemy_char)),
                               dim=-1).float()
            out = self.lstm_info(i_info, prev_lstm_info_state)
            lstm_info_output, lstm_info_state = out

            i_controls = torch.cat((lstm_info_output,
                                    state[:, :, 10:14],
                                    self.embd_inputs(prev_input),
                                    self.embd_chars(player_char)),
                                   dim=-1).float()
            out = self.lstm_controls(i_controls, prev_lstm_controls_state)
            lstm_controls_output, lstm_controls_state = out

            lstm_outputs = torch.cat((lstm_info_output,
                                      lstm_controls_output),
                                     dim=-1).float()
            p_actions = self.fc_net(lstm_outputs)

        return(p_actions, lstm_info_state, lstm_controls_state)


class MorkomAI:
    """Class used for training and using the LSTM network.

    Attributes
    ----------
    net: MKNet
        The neural network being used.
    model_folder: str
        The path to the folder to save/load model checkpoints from.
    use_cuda: bool
        Whether to use CUDA for tensor calculations.
    exploration_rate: float
        Rate of taking random actions as opposed to using network outputs.
    exploration_rate_decay: float
        The exploration rate gets multiplied by this value after each
        action.
    exploration_rate_min: float
        The minimum exploration rate.
    curr_step: int
        The number of steps taken so far.
    n_rounds: int
        The number of rounds fought, not updated internally.
    save_every: int
        The number of steps to take between saving the model.
    burinin: int
        The number of steps to take before starting to learn.
    learn_every: int
        The number of steps to take between learning from memory.
    sync_every: int
        The number of steps to take between syncing the target network
        parameters to the online network parameters.
    memory: list
        A list containing all the previous transitions.
    max_memory: int
        The maximum size of memory list. Older values are forgotten once this
        is exceeded.
    batch_size: int
        The number of transition states to sample when learning.
    max_sequence_length: int
        The maximum sequence size to learn.
    gamma: float
        The discount rate of TD target.
    discounts: tuple
        The reward discounts for sequences.
    optimizer: torch.optim.Adam
        The network optimizer.
    loss_fn: torch.nn.SmoothL1Loss
        The loss function used for optimization.

    Parameters
    ----------
    model_folder: str
        The path to the folder to save/load model checkpoints from.
    """
    def __init__(self, model_folder: str) -> None:
        self.model_folder = model_folder

        self.use_cuda = torch.cuda.is_available()
        self.net = MKNet().float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.05
        self.curr_step = 0
        self.n_rounds = 0

        self.save_every = 5000
        self.burnin = 500
        self.learn_every = 25
        self.sync_every = 5000

        self.memory = []
        self.max_memory = 100000
        self.batch_size = 32
        self.max_sequence_length = 10
        self.gamma = 0.9

        self.set_discounts()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.load_most_recent()

    def set_discounts(self) -> None:
        """Sets the discounts for rewards."""
        discounts = [0 for _ in range(self.max_sequence_length)]
        for i in range(self.max_sequence_length):
            discounts[i] = self.gamma ** (i)
        self.discounts = tuple(discounts)

    def get_random_states(self) -> tuple:
        """Create random state tuples for the LSTMs.

        Returns
        -------
        lstm_info_state: tuple
           A tuple of 2 random tensors corresponding to the state tensors of
           net.lstm_info.
        lstm_controls_state: tuple
           A tuple of 2 random tensors corresponding to the state tensors of
           net.lstm_controls.
        """
        # (number of layers, batch size, hidden size)
        dims_char = (self.net.lstm_info_opt['num_layers'], 1,
                     self.net.lstm_info_opt['hidden_size'])
        lstm_info_state = (torch.randn(*dims_char), torch.randn(*dims_char))

        dims_enemy = (self.net.lstm_controls_opt['num_layers'], 1,
                      self.net.lstm_controls_opt['hidden_size'])
        lstm_controls_state = (torch.randn(*dims_enemy),
                               torch.randn(*dims_enemy))

        return(lstm_info_state, lstm_controls_state)

    def act(self, state: torch.tensor,
            lstm_info_state: tuple,
            lstm_controls_state: tuple) -> tuple:
        """Step the model forward and choose an action.

        Parameters
        ----------
        state: torch.tensor
            Current game state tensor.
        lstm_info_state: tuple
            The previous state of lstm_info.
        lstm_controls_state: tuple
            The previous state of lstm_info.

        Returns
        -------
        action: int
            Index of action to take.
        lstm_info_state: tuple
            The new state of lstm_info.
        lstm_controls_state: tuple
            The new state of lstm_info.
        """
        n_actions = self.net.fc_net_opt[-1]['out_features']
        # EXPLOIT
        net_returns = self.net(state, lstm_info_state, lstm_controls_state,
                               online=True)
        p_actions, lstm_info_state, lstm_controls_state = net_returns
        p_actions = p_actions.flatten()
        action = random.choices(list(range(n_actions)), p_actions, k=1)[0]

        # EXPLORE
        if random.random() < self.exploration_rate:
            action = np.random.randint(n_actions)

        # Decrease exploration_rate
        new_rate = self.exploration_rate * self.exploration_rate_decay
        if new_rate > self.exploration_rate_min:
            self.exploration_rate = new_rate
        else:
            self.exploration_rate = self.exploration_rate_min

        # Increment step
        self.curr_step += 1
        return(action, lstm_info_state, lstm_controls_state)

    def cache(self, state: torch.tensor, lstm_info_state: torch.tensor,
              lstm_controls_state: torch.tensor, next_state: torch.tensor,
              next_lstm_info_state: torch.tensor,
              next_lstm_controls_state: torch.tensor,
              action: int, reward: float) -> None:
        """Store state transition to memory.

        Parameters
        ----------
        state: torch.tensor
            Game state tensor.
        lstm_info_state: tuple
            The lstm_info state tensors.
        lstm_controls_state: tuple
            The lstm_info state tensors.
        next_state: torch.tensor
            Game state tensor at the next state.
        next_lstm_info_state: tuple
            The lstm_info state tensors at the next state.
        next_lstm_controls_state: tuple
            The lstm_info state tensors at the next state.
        action: int
            The action taken.
        reward: float
            The reward for taking that action.
        """
        state = state.squeeze(0).detach()
        lstm_info_state = tuple(x.squeeze(1).detach()
                                for x in lstm_info_state)
        lstm_controls_state = tuple(x.squeeze(1).detach()
                                    for x in lstm_controls_state)

        next_state = state.squeeze(0).detach()
        next_lstm_info_state = tuple(x.squeeze(1).detach()
                                     for x in next_lstm_info_state)
        next_lstm_controls_state = tuple(x.squeeze(1).detach()
                                         for x in next_lstm_controls_state)

        action = torch.tensor([action]).detach()
        reward = torch.tensor([reward]).detach()

        if self.use_cuda:
            state = state.cuda()
            lstm_info_state = lstm_info_state.cuda()
            lstm_controls_state = lstm_controls_state.cuda()
            next_state = next_state.cuda()
            next_lstm_info_state = next_lstm_info_state.cuda()
            next_lstm_controls_state = next_lstm_controls_state.cuda()
            action = action.cuda()
            reward = reward.cuda()

        self.memory.append([state, lstm_info_state, lstm_controls_state,
                            next_state, next_lstm_info_state,
                            next_lstm_controls_state, action, reward])

    def recall(self, trailing_samples: int = 0) -> list:
        """Retrieve a batch of experiences from memory

        Parameters
        ----------
        trailing_samples: int, optional
            The number of trailing samples to include. Must be an integer
            between 0 and (self.max_sequence_length - 1). Defaults to 0.

        Returns
        -------
        state: torch.tensor
            Game state tensor.
        lstm_info_state: tuple
            The lstm_info state tensors.
        lstm_controls_state: tuple
            The lstm_info state tensors.
        next_state: torch.tensor
            Game state tensor at the next state.
        next_lstm_info_state: tuple
            The lstm_info state tensors at the next state.
        next_lstm_controls_state: tuple
            The lstm_info state tensors at the next state.
        action: int
            The action taken.
        reward: float
            The reward for taking that action.
        """
        if not (0 <= trailing_samples < self.max_sequence_length):
            raise ValueError(f'Invalid trailing samples: {trailing_samples}')

        samples_remaining = self.batch_size
        batch = []
        while samples_remaining > 0:
            samples = random.sample(range(trailing_samples, len(self.memory)),
                                    samples_remaining)
            new_batch = [self.memory[i-trailing_samples:i+1] for i in samples]
            samples_remaining = 0
            # Ensure that the sequence is for a single character, there can
            # still be some spillover across rounds/matches but this prevents
            # training the wrong lstm character network
            for sequence in new_batch:
                batch.append(sequence)  # Add to batch
                character = sequence[0][0][0, 16]
                for cached_value in sequence:
                    if character != cached_value[0][0, 16]:
                        # Remove sequence from batch and require 1 new sample
                        batch.pop()
                        samples_remaining += 1
                        break

        lstm_info_dims = (self.net.lstm_info_opt['num_layers'],
                          self.batch_size,
                          self.net.lstm_info_opt['hidden_size'])
        lstm_controls_dims = (self.net.lstm_info_opt['num_layers'],
                              self.batch_size,
                              self.net.lstm_controls_opt['hidden_size'])

        state = torch.zeros((self.batch_size, trailing_samples + 1, 19))
        lstm_info_state = (torch.zeros(lstm_info_dims),
                           torch.zeros(lstm_info_dims))
        lstm_controls_state = (torch.zeros(lstm_controls_dims),
                               torch.zeros(lstm_controls_dims))
        next_state = torch.zeros((self.batch_size, trailing_samples + 1, 19))
        next_lstm_info_state = (torch.zeros(lstm_info_dims),
                                torch.zeros(lstm_info_dims))
        next_lstm_controls_state = (torch.zeros(lstm_controls_dims),
                                    torch.zeros(lstm_controls_dims))
        action = torch.zeros((self.batch_size, 1))
        reward = torch.zeros((self.batch_size, 1))

        for i, sequence in enumerate(batch):
            total_reward = 0
            for j, cached_value in enumerate(sequence):
                # Find total reward for sequence
                total_reward += cached_value[7].item() * self.discounts[-(j+1)]
                state[i, j, :] = cached_value[0]
                next_state[i, j, :] = cached_value[3]
            # Set states to first in sequence
            # On left side, 0/1 indicates state index in tuple, [:, i, :] used
            # since batch is second dimension
            # On right side, 0 is first in sequqnce, next index is the state
            # tuple, and for the last index 0/1 indicates state index in tuple
            lstm_info_state[0][:, i, :] = sequence[0][1][0]
            lstm_info_state[1][:, i, :] = sequence[0][1][1]
            lstm_controls_state[0][:, i, :] = sequence[0][2][0]
            lstm_controls_state[1][:, i, :] = sequence[0][2][1]
            next_lstm_info_state[0][:, i, :] = sequence[0][4][0]
            next_lstm_info_state[1][:, i, :] = sequence[0][4][1]
            next_lstm_controls_state[0][:, i, :] = sequence[0][5][0]
            next_lstm_controls_state[1][:, i, :] = sequence[0][5][1]
            action[i] = cached_value[6]
            reward[i] = total_reward

        return(state, lstm_info_state, lstm_controls_state,
               next_state, next_lstm_info_state, next_lstm_controls_state,
               action, reward)

    def td_estimate(self, state: torch.tensor,
                    lstm_info_state: tuple,
                    lstm_controls_state: tuple,
                    action: torch.tensor) -> torch.tensor:
        """The prediction from the current state.

        Parameters
        ----------
        state: torch.tensor
            Game state tensor.
        lstm_info_state: tuple
            The lstm_info state tensors.
        lstm_controls_state: tuple
            The lstm_info state tensors.
        action: torch.tensor
            The actions taken.

        Returns
        -------
        current_Q: torch.tensor
            The probability of taking the given action for each sample in the
            batch.
        """
        if self.use_cuda:
            state = state.cuda()
            lstm_info_state = (x.cuda() for x in lstm_info_state)
            lstm_controls_state = (x.cuda() for x in lstm_controls_state)
            action = action.cuda()

        # Q_online(s,a)
        current_Q = self.net(state, lstm_info_state, lstm_controls_state, True
                             )[0][np.arange(0, self.batch_size),
                                  -1, action.squeeze().long()]
        return(current_Q)

    @torch.no_grad()
    def td_target(self, reward: torch.tensor,
                  next_state: torch.tensor,
                  next_lstm_info_state: tuple,
                  next_lstm_controls_state: tuple) -> torch.tensor:
        """Current reward and the estimated Q* in the next state.

        Parameters
        ----------
        reward: torch.tensor
            The reward for the given transitions.
        next_state: torch.tensor
            Game state tensor for next state.
        next_lstm_info_state: tuple
            The lstm_info state tensors for next state.
        next_lstm_controls_state: tuple
            The lstm_info state tensors for next state.

        Returns
        -------
        TD_t: torch.tensor
            TD target.
        """
        next_state_Q = self.net(next_state,
                                next_lstm_info_state,
                                next_lstm_controls_state, True)
        best_action = torch.argmax(next_state_Q[0], axis=2)[:, -1]

        next_Q = self.net(next_state,
                          next_lstm_info_state,
                          next_lstm_controls_state,
                          False)[0][np.arange(0, self.batch_size),
                                    -1, best_action]
        return((reward.squeeze() + self.gamma * next_Q).float())

    def update_Q_online(self, td_estimate: torch.tensor,
                        td_target: torch.tensor) -> float:
        """Update the online network.

        Parameters
        ----------
        td_estimate: torch.tensor
            The predicted optimal Q* for a given state s.
        td_target: torch.tensor
            Aggregation of current reward and the estimated Q* in the next
            state s'.

        Returns
        -------
        loss: float
            The output of the loss function.
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return(loss.item())

    def sync_Q_target(self) -> None:
        """Sync target network parameters to the online network parameters."""
        embd_action_state_dict = self.net.embd_actions_online.state_dict()
        embd_inputs_state_dict = self.net.embd_inputs_online.state_dict()
        embd_chars_state_dict = self.net.embd_chars_online.state_dict()
        lstm_info_state_dict = self.net.lstm_info_online.state_dict()
        lstm_controls_state_dict = self.net.lstm_controls_online.state_dict()
        fc_net_state_dict = self.net.fc_net_online.state_dict()

        self.net.embd_actions.load_state_dict(embd_action_state_dict)
        self.net.embd_inputs.load_state_dict(embd_inputs_state_dict)
        self.net.embd_chars.load_state_dict(embd_chars_state_dict)
        self.net.lstm_info.load_state_dict(lstm_info_state_dict)
        self.net.lstm_controls.load_state_dict(lstm_controls_state_dict)
        self.net.fc_net.load_state_dict(fc_net_state_dict)

    def save(self) -> None:
        """Save model to disk."""
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]
        file_name = f"{int(self.curr_step // self.save_every)}.chkpt"
        save_path = f"{self.model_folder}/{file_name}"
        torch.save(dict(model=self.net.state_dict(),
                        exploration_rate=self.exploration_rate,
                        curr_step=self.curr_step,
                        memory=self.memory,
                        n_rounds=self.n_rounds), save_path)
        print(f"Network saved to {save_path} at step {self.curr_step}.")

    def load(self, path: str) -> None:
        """Load a model from disk.

        Parameters
        ----------
        path: str
            The path to the model file.
        """
        print(f'Loading model checkpoint from "{path}"')
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.curr_step = checkpoint['curr_step']
        self.memory = checkpoint['memory']
        self.n_rounds = checkpoint['n_rounds']

    def load_most_recent(self) -> None:
        """Loads the last saved model on disk."""
        try:
            files = glob.glob(f'{self.model_folder}/*.chkpt')
            latest_file = max(files, key=os.path.getctime)
            self.load(latest_file)
        except ValueError:
            print('Could not load latest checkpoint.')  # No files in directory

    def learn(self) -> tuple:
        """Update the network models.

        Returns
        -------
        td_est_mean: float | None
            The mean td_estimate value. Returns None if no update was made.
        loss: float | None
            The loss function ouput. Returns None if no update was made.
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return(None, None)

        if self.curr_step % self.learn_every != 0:
            return(None, None)

        # Sample from memory
        seq_length = random.randint(0, self.max_sequence_length - 1)
        sample = self.recall(seq_length)
        state = sample[0]
        lstm_info_state = sample[1]
        lstm_controls_state = sample[2]
        next_state = sample[3]
        next_lstm_info_state = sample[4]
        next_lstm_controls_state = sample[5]
        action = sample[6]
        reward = sample[7]

        # Get TD Estimate
        td_est = self.td_estimate(state, lstm_info_state, lstm_controls_state,
                                  action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, next_lstm_info_state,
                                next_lstm_controls_state)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return(td_est.mean().item(), loss)
