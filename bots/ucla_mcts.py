import math
import random
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_interface import BaseAgent
from bots.ucla_bot_heuristic import UCLAHeuristicEvaluator

class Node:
    __slots__ = ('parent', 'move', 'children', 'visits', 'value_sum', 'player_to_move', 'untried_moves', 'terminal')

    def __init__(self, parent, move, player_to_move, valid_moves, terminal):
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.player_to_move = player_to_move
        self.terminal = terminal
        self.untried_moves = valid_moves if valid_moves is not None else []

class UCLAMCTS(BaseAgent):
    """
    MCTS that uses the UCLAHeuristicEvaluator for leaf node evaluation 
    instead of random rollouts or neural networks.
    """
    def __init__(self, name: str = "UCLA MCTS", n_simulations: int = 1000, time_limit: float = 2.0, c_puct: float = 1.5, scale: float = 2000.0):
        super().__init__(name)
        self.n_simulations = n_simulations
        self.time_limit = time_limit
        self.c_puct = c_puct
        self.scale = scale
        self.evaluator = UCLAHeuristicEvaluator()

    def get_move(self, game_state):
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return -1
            
        # Order the initial moves using the heuristic evaluator
        ordered_valid = self.evaluator.order_moves(game_state, valid_moves)
            
        root = Node(parent=None, move=None, player_to_move=game_state.current_player, 
                    valid_moves=ordered_valid,
                    terminal=not game_state.is_running())
                    
        simulations_done = 0
        start_time = time.time()
        
        while True:
            if self.time_limit is not None:
                if time.time() - start_time >= self.time_limit:
                    break
            elif simulations_done >= self.n_simulations:
                break
                
            simulations_done += 1
            
            node = root
            moves_made = 0
            
            # SELECTION
            while not node.untried_moves and node.children:
                node = self._select_child(node)
                game_state.execute_move(node.move)
                moves_made += 1
                
            # EXPANSION
            if node.untried_moves:
                move = node.untried_moves.pop(0)
                game_state.execute_move(move)
                moves_made += 1
                
                terminal = not game_state.is_running()
                if not terminal:
                    valid_for_child = self.evaluator.order_moves(game_state, game_state.get_valid_moves())
                else:
                    valid_for_child = []
                    
                child = Node(parent=node, move=move, player_to_move=game_state.current_player,
                             valid_moves=valid_for_child,
                             terminal=terminal)
                node.children.append(child)
                node = child
                
            # LEAF EVALUATION (using heuristic instead of rollout)
            if game_state.is_running():
                raw_val = self.evaluator.evaluate(game_state, player=node.player_to_move)
                # Normalize value to roughly [-1, 1] using tanh.
                # UCLA heuristic gives roughly 1000 per box difference.
                val = math.tanh(raw_val / self.scale)
            else:
                # terminal
                my_boxes = int((game_state.b == node.player_to_move).sum())
                opp_boxes = int((game_state.b == -node.player_to_move).sum())
                if my_boxes > opp_boxes:
                    val = 1.0
                elif my_boxes < opp_boxes:
                    val = -1.0
                else:
                    val = 0.0
                
            # BACKPROPAGATION
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value_sum += val
                
                if curr.parent is not None:
                    # If the parent's turn player is different, the value is inverted
                    if curr.parent.player_to_move != curr.player_to_move:
                        val = -val
                
                curr = curr.parent
                
            # UNDO
            for _ in range(moves_made):
                game_state.undo_move()
                
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.move
        return ordered_valid[0]

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        log_n = math.log(node.visits) if node.visits > 0 else 0
        
        for child in node.children:
            if child.visits == 0:
                ucb = float('inf')
            else:
                q = child.value_sum / child.visits
                if node.player_to_move != child.player_to_move:
                    q = -q
                ucb = q + self.c_puct * math.sqrt(log_n / child.visits)
            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child
