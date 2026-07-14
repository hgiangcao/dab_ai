import argparse
import os
import random
import numpy as np
import torch

from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict


def play_episode(nnet, game_size, max_steps=200, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    game = DotsAndBoxesGame(size=game_size, starting_player=1)
    episode = []
    step = 0

    while game.is_running() and step < max_steps:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break

        state = nnet.format_game_state(game)
        probs, _ = nnet.predict(state)
        valid_probs = np.zeros_like(probs, dtype=np.float64)
        valid_probs[valid_moves] = probs[valid_moves]
        total = valid_probs.sum()
        if total <= 0:
            action = valid_moves[0]
        else:
            valid_probs = valid_probs / total
            action = int(np.random.choice(valid_moves, p=valid_probs[valid_moves]))

        game.execute_move(action)
        episode.append((state, action, 0.0))
        step += 1

    result = game.result if game.result is not None else 0
    reward = 1.0 if result == 1 else -1.0 if result == -1 else 0.0
    return [(state, action, reward) for state, action, _ in episode]


def train_actor_critic(game_size=2, iterations=5, episodes_per_iter=8, seed=0, checkpoint_dir='checkpoints/ac'):
    args = dotdict({
        'lr': 1e-3,
        'epochs': 1,
        'batch_size': 8,
        'num_channels': 8,
        'num_res_blocks': 1,
        'l2_reg': 1e-4,
        'device': 'cpu',
    })

    game = DotsAndBoxesGame(size=game_size)
    nnet = NNetWrapper(game, args)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(checkpoint_dir, exist_ok=True)

    for iteration in range(iterations):
        trajectories = []
        for ep_idx in range(episodes_per_iter):
            trajectories.append(play_episode(nnet, game_size, seed=seed + iteration * 100 + ep_idx))

        metrics = nnet.train_actor_critic(trajectories, gamma=0.99, entropy_weight=0.01)
        print(f"iteration={iteration + 1} policy_loss={metrics[0]:.4f} value_loss={metrics[1]:.4f} total_loss={metrics[2]:.4f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"ac_iter_{iteration + 1}.pth.tar")
        nnet.save_checkpoint(folder=checkpoint_dir, filename=os.path.basename(checkpoint_path))

    return nnet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a simple actor-critic agent for Dots and Boxes without MCTS')
    parser.add_argument('--game-size', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--episodes-per-iter', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/ac')
    args = parser.parse_args()

    train_actor_critic(
        game_size=args.game_size,
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )
