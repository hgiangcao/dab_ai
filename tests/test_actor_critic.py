import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict


def test_actor_critic_training_runs_on_a_single_episode():
    args = dotdict({
        'lr': 1e-3,
        'epochs': 1,
        'batch_size': 2,
        'num_channels': 8,
        'num_res_blocks': 1,
        'l2_reg': 1e-4,
        'device': 'cpu',
    })

    game = DotsAndBoxesGame(size=2)
    nnet = NNetWrapper(game, args)

    episode = []
    local_game = DotsAndBoxesGame(size=2, starting_player=1)
    while local_game.is_running():
        valid_moves = local_game.get_valid_moves()
        if not valid_moves:
            break
        move = valid_moves[0]
        state = nnet.format_game_state(local_game)
        episode.append((state, move, 0.0))
        local_game.execute_move(move)

    metrics = nnet.train_actor_critic([episode], gamma=0.9, entropy_weight=0.01)
    assert isinstance(metrics, tuple)
    assert len(metrics) == 3
    assert all(np.isfinite(v) for v in metrics)
