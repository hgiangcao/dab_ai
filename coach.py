import os
import copy
import random
import numpy as np
from collections import deque
from mcts import MCTS

class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.nnet, self.args)
        self.train_examples_history = [] 

    def execute_episode(self, fill_percentage=0.0):
        """
        Executes one episode of self-play.
        fill_percentage: Used for backward training. Generates a partially filled board.
        """
        train_examples = []
        board = copy.deepcopy(self.game)

        # Backward Training setup: Pre-fill the board with random valid moves
        if fill_percentage > 0.0:
            total_moves = int(board.N_LINES * fill_percentage)
            for _ in range(total_moves):
                valid_moves = board.get_valid_moves()
                if not valid_moves:
                    break
                board.execute_move(random.choice(valid_moves))
        
        # In AlphaZero, MCTS tree is preserved/reset per episode, not per turn.
        self.mcts = MCTS(self.nnet, self.args)
        
        step = 0
        while True:
            step += 1
            temp = int(step < self.args.temp_threshold)
            
            # 1. Run MCTS to get policy probabilities
            pi = self.mcts.play(board, temp=temp)
            
            # 2. Format current state into 4-channel tensor for training memory
            h, v = board.l_to_h_v(board.get_canonical_lines())
            p1_boxes = np.where(board.get_canonical_boxes() == 1, 1.0, 0.0)
            p2_boxes = np.where(board.get_canonical_boxes() == -1, 1.0, 0.0)
            stacked_board = np.stack([h[:-1, :], v[:, :-1], p1_boxes, p2_boxes], axis=0)
            
            # 3. Store the state, current player, and policy
            train_examples.append([stacked_board, board.current_player, pi, None])
            
            # 4. Pick a move and execute
            action = np.random.choice(len(pi), p=pi)
            board.execute_move(action)
            
            # 5. Check if game is over
            if not board.is_running():
                # Assign actual game outcome to all recorded moves
                result = board.result
                return [(
                    x[0], 
                    x[2], 
                    1.0 if x[1] == result else (-1.0 if result != 0 else 0.0)
                ) for x in train_examples]

    def learn(self):
        """
        Main training loop: Self-play -> Training -> Iteration
        """
        for i in range(1, self.args.num_iters + 1):
            print(f"--- Iteration {i}/{self.args.num_iters} ---")
            
            iteration_train_examples = deque([], maxlen=self.args.maxlen_queue)
            
            # Calculate fill percentage for backward training phase
            fill_pct = max(0.0, self.args.start_fill_pct - (i * self.args.fill_decay))
            
            for eps in range(self.args.num_eps):
                iteration_train_examples += self.execute_episode(fill_percentage=fill_pct)
                
            self.train_examples_history.append(iteration_train_examples)
            
            if len(self.train_examples_history) > self.args.keep_history_iters:
                self.train_examples_history.pop(0)
                
            # Flatten history
            train_data = []
            for e in self.train_examples_history:
                train_data.extend(e)
            
            # Train the neural network
            self.nnet.train(train_data)
            
            # Save checkpoint
            self.nnet.save_checkpoint(folder=self.args.checkpoint_dir, filename=f'checkpoint_{i}.pth.tar')