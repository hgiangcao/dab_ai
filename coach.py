import os
import numpy as np
from collections import deque
from tqdm import tqdm
from game import DotsAndBoxesGame

def play_match(game, p1_agent, p2_agent, num_games):
    """Arena gatekeeper logic to compare two agents."""
    p1_wins, p2_wins, draws = 0, 0, 0
    half_games = num_games // 2

    def play_single(agent1, agent2):
        players = {1: agent1, -1: agent2}
        cur_player = 1
        board = game.getInitBoard()
        while game.getGameEnded(board, cur_player) == 0:
            canon_board = game.getCanonicalForm(board, cur_player)
            action = players[cur_player](canon_board)
            board, cur_player = game.getNextState(board, cur_player, action)
        return cur_player * game.getGameEnded(board, cur_player)

    for _ in range(half_games):
        res = play_single(p1_agent, p2_agent)
        if res == 1: p1_wins += 1
        elif res == -1: p2_wins += 1
        else: draws += 1

    for _ in range(num_games - half_games):
        res = play_single(p2_agent, p1_agent)
        if res == -1: p1_wins += 1
        elif res == 1: p2_wins += 1
        else: draws += 1

    return p1_wins, p2_wins, draws

class AlphaZeroTrainer:
    def __init__(self, game, nnet, pnet, mcts_class, args):
        self.game = game
        self.nnet = nnet
        self.pnet = pnet
        self.MCTS = mcts_class
        self.args = args
        self.memory = deque(maxlen=self.args.maxlen_queue)

    def execute_episode(self):
        """Generates self-play data and applies symmetry augmentation."""
        train_examples = []
        board = self.game.getInitBoard()
        cur_player = 1
        episode_step = 0
        
        mcts = self.MCTS(self.game, self.nnet, self.args)

        while True:
            episode_step += 1
            canonical_board = self.game.getCanonicalForm(board, cur_player)
            temp = int(episode_step < self.args.temp_threshold)
            
            pi = mcts.getActionProb(canonical_board, temp=temp)
            sym_examples = self.game.getSymmetries(canonical_board, pi)
            
            for sym_board, sym_pi in sym_examples:
                train_examples.append([sym_board, cur_player, sym_pi, None])

            action = np.random.choice(len(pi), p=pi)
            board, cur_player = self.game.getNextState(board, cur_player, action)
            r = self.game.getGameEnded(board, cur_player)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples]

    def learn(self):
        """
        Training loop matching the continuous-update and baseline-evaluation approach.
        """
        starting_iteration = 1
        evaluation_results = { 'Random': [] } 

        for i in range(starting_iteration, self.args.num_iters + 1):
            print(f"\n#################### Iteration {i}/{self.args.num_iters} ####################")
            
            # 1. Self-Play
            print("------------ Self-Play ------------")
            iteration_data = []
            for _ in tqdm(range(self.args.num_eps), desc="Self Play"):
                iteration_data.append(self.execute_episode())
            
            # 2. Augment Data & Add to Memory
            augmented_data = self.augment_data(iteration_data)
            flat_data = [example for game in augmented_data for example in game]
            self.memory.extend(flat_data)
            
            # 3. Neural Network Training (Delegated to NNetWrapper)
            print("\n---------- Neural Network Training -----------")
            self.nnet.train(list(self.memory))
            
            # 4. Model Comparison 
            print("\n-------------- Model Comparison --------------")
            nmcts = self.MCTS(self.game, self.nnet, self.args)
            neural_net_agent = lambda x: np.argmax(nmcts.getActionProb(x, temp=0))
            
            random_agent = lambda x: np.random.choice(np.where(self.game.getValidMoves(x, 1) == 1)[0])
            
            wins, losses, draws = play_match(self.game, neural_net_agent, random_agent, self.args.arena_games)
            evaluation_results['Random'].append((wins, losses, draws))
            print(f"Vs Random -> Wins: {wins} | Losses: {losses} | Draws: {draws}")
            
            # 5. Save Checkpoints (Delegated to NNetWrapper)
            print("\nSaving Checkpoint...")
            self.nnet.save_checkpoint(folder=self.args.checkpoint_dir, filename=f'checkpoint_{i}.pth.tar')
            self.nnet.save_checkpoint(folder=self.args.checkpoint_dir, filename='best.pth.tar')

    @staticmethod
    def augment_data(train_examples_per_game: list):
        data_augmented = []
        for train_examples in train_examples_per_game:
            train_examples_augmented = []
            for lines, boxes, p, v in train_examples:
                train_examples_augmented.extend(zip(
                    DotsAndBoxesGame.get_rotations_and_reflections_lines(lines),
                    DotsAndBoxesGame.get_rotations_and_reflections_boxes(boxes),
                    DotsAndBoxesGame.get_rotations_and_reflections_lines(np.asarray(p)),
                    [v] * 8
                ))
            data_augmented.append(train_examples_augmented)

        return data_augmented