import numpy as np
from game import DotsAndBoxesGame # Your engine

class DotsAndBoxesAlphaZeroWrapper:
    """
    Adapter class to make your engine compatible with alpha-zero-general pipelines.
    """
    def __init__(self, size=5):
        self.size = size
        self.action_size = 2 * size * (size + 1)

    def getInitBoard(self):
        # Returns the initial board state (empty arrays)
        game = DotsAndBoxesGame(self.size)
        return (game.l, game.b) # Return the state representation

    def getBoardSize(self):
        # Neural network input dimensions
        return (self.size, self.size)

    def getActionSize(self):
        # Total number of possible lines
        return self.action_size

    def getNextState(self, board_state, player, action):
        # 1. Instantiate a game engine with the given state
        game = DotsAndBoxesGame(self.size, starting_player=player)
        game.l, game.b = board_state[0].copy(), board_state[1].copy()
        
        # 2. Execute the action
        game.execute_move(action)
        
        # 3. Return the new state and whose turn it is now
        return (game.l, game.b), game.current_player

    def getValidMoves(self, board_state, player):
        # Return a binary vector of length action_size (1 = valid, 0 = invalid)
        valid_moves = np.zeros(self.action_size)
        valid_indices = np.where(board_state[0] == 0)[0]
        valid_moves[valid_indices] = 1
        return valid_moves

    def getGameEnded(self, board_state, player):
        # Return 0 if not ended, 1 if player won, -1 if player lost, a small float for draw
        game = DotsAndBoxesGame(self.size)
        game.l, game.b = board_state[0], board_state[1]
        
        # You need to calculate the score manually here if result isn't cached
        p1_score = np.sum(game.b == 1)
        p2_score = np.sum(game.b == -1)
        total_boxes = self.size * self.size
        
        if p1_score + p2_score < total_boxes:
            return 0 # Game is still running
            
        if p1_score > p2_score:
            return 1 if player == 1 else -1
        elif p2_score > p1_score:
            return 1 if player == -1 else -1
        else:
            return 1e-4 # Draw