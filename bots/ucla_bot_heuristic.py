import numpy as np
import random
import sys
import os
# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_interface import BaseAgent

class UCLAHeuristicEvaluator:
    """
    Heuristic evaluator extracted from UCLABot_v3.
    Designed for:
        AlphaBeta
        Negamax
        MCTS rollout evaluation
    Positive value = advantage for current player
    """
    def __init__(self):
        self.W_SCORE = 1000
        self.W_CHAIN = 80
        self.W_SAFE = 20
        self.W_DANGER = 120
        self.W_MOBILITY = 5
    # ==========================================================
    # Main evaluation
    # ==========================================================
    def evaluate(self, game, player=None):
        if player is None:
            player = game.current_player
        opponent = -player
        value = 0
        # ----------------------------------
        # 1. Captured boxes
        # ----------------------------------
        my_boxes = np.sum(game.b == player)
        opp_boxes = np.sum(game.b == opponent)
        value += (
            my_boxes - opp_boxes
        ) * self.W_SCORE
        # ----------------------------------
        # 2. Box danger
        # ----------------------------------
        box_state = self.get_box_state(game)
        my_three = 0
        opp_three = 0
        for r in range(game.SIZE):
            for c in range(game.SIZE):
                if box_state[r][c] == 3:
                    # estimate owner who will face it
                    if self.box_owner_risk(
                            game,
                            r,
                            c,
                            player):
                        my_three += 1
                    else:
                        opp_three += 1
        value += (
            opp_three -
            my_three
        ) * self.W_DANGER
        # ----------------------------------
        # 3. Chain structure
        # ----------------------------------
        chains = self.find_chains(game)
        for chain in chains:
            length = len(chain)
            if length >= 3:
                value += (
                    self.W_CHAIN *
                    self.chain_control(
                        game,
                        chain,
                        player
                    )
                )
        # ----------------------------------
        # 4. Safe moves
        # ----------------------------------
        safe = self.count_safe_moves(game)
        value += (
            safe *
            self.W_SAFE
        )
        # ----------------------------------
        # 5. Mobility
        # ----------------------------------
        value += (
            len(game.get_valid_moves())
            *
            self.W_MOBILITY
        )
        return float(value)
    # ==========================================================
    # Board parsing
    # ==========================================================
    def get_box_state(self,game):
        result=np.zeros(
            (game.SIZE,game.SIZE),
            dtype=int
        )
        for r in range(game.SIZE):
            for c in range(game.SIZE):
                lines = game.get_lines_of_box(
                    (r,c)
                )
                result[r][c]=np.count_nonzero(
                    game.l[lines]
                )
        return result
    # ==========================================================
    # Safe move detection
    # ==========================================================
    def is_safe_move(self,game,line):
        affected=game.get_boxes_of_line(line)
        for box in affected:
            lines=game.get_lines_of_box(box)
            count=np.count_nonzero(
                game.l[lines]
            )
            if count==3:
                return False
        return True
    def count_safe_moves(self,game):
        count=0
        for m in game.get_valid_moves():
            if self.is_safe_move(game,m):
                count+=1
        return count
    # ==========================================================
    # Chain detection
    # ==========================================================
    def find_chains(self,game):
        state=self.get_box_state(game)
        visited=set()
        chains=[]
        for r in range(game.SIZE):
            for c in range(game.SIZE):
                if (
                    state[r][c]>=2
                    and
                    (r,c) not in visited
                ):
                    chain=[]
                    self.dfs_chain(
                        state,
                        r,
                        c,
                        visited,
                        chain
                    )
                    if len(chain)>1:
                        chains.append(chain)
        return chains
    def dfs_chain(
            self,
            state,
            r,
            c,
            visited,
            chain):
        if (
            r<0 or
            c<0 or
            r>=state.shape[0] or
            c>=state.shape[1]
        ):
            return
        if (r,c) in visited:
            return
        if state[r][c]<2:
            return
        visited.add((r,c))
        chain.append((r,c))
        dirs=[
            (-1,0),
            (1,0),
            (0,-1),
            (0,1)
        ]
        for dr,dc in dirs:
            self.dfs_chain(
                state,
                r+dr,
                c+dc,
                visited,
                chain
            )
    def chain_control(
            self,
            game,
            chain,
            player):
        """
        Estimate who benefits from chain.
        +1 = player controls
        -1 = opponent controls
        """
        length=len(chain)
        # odd chains are usually favorable
        # to the player who opens them
        if length%2==1:
            return 1
        return -1
    # ==========================================================
    # Move ordering for AlphaBeta
    # ==========================================================
    def move_priority(self,game,move):
        priority=0
        # capture test
        boxes_before=np.sum(game.b!=0)
        game.execute_move(move)
        boxes_after=np.sum(game.b!=0)
        game.undo_move()
        captured=boxes_after-boxes_before
        if captured:
            priority += (
                10000 +
                captured*1000
            )
        # safe moves first
        if self.is_safe_move(game,move):
            priority+=5000
        else:
            priority-=5000
        return priority
    def order_moves(self,game,moves):
        return sorted(
            moves,
            key=lambda x:
                self.move_priority(game,x),
            reverse=True
        )
    # ==========================================================
    # Risk estimation
    # ==========================================================
    def box_owner_risk(
            self,
            game,
            r,
            c,
            player):
        """
        Estimate whether player
        will be forced to open box.
        """
        lines=game.get_lines_of_box(
            (r,c)
        )
        empty=sum(
            game.l[x]==0
            for x in lines
        )
        # only one remaining edge
        if empty==1:
            return (
                game.current_player==player
            )
        return False

class UCLAGreedyBot(BaseAgent):
    """
    A simple 1-step lookahead greedy bot that evaluates all valid moves 
    using UCLAHeuristicEvaluator and picks the best one.
    """
    def __init__(self, name="UCLAGreedy"):
        super().__init__(name)
        self.evaluator = UCLAHeuristicEvaluator()
    def get_move(self, game) -> int:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return -1
        best_score = float('-inf')
        best_moves = []
        
        our_player = game.current_player
        
        # Use evaluator's move_priority to sort moves, which speeds up AlphaBeta 
        # but here we can just use it to resolve ties, or simply evaluate all.
        for move in valid_moves:
            game.execute_move(move)
            score = self.evaluator.evaluate(game, player=our_player)
            game.undo_move()
            
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
                
        # Among equally scored moves, maybe prefer those with higher move_priority?
        # Actually, let's just pick one randomly to break ties.
        return random.choice(best_moves)