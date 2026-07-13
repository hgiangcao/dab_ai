import random
from agent_interface import BaseAgent

class GreedyChainPlayer(BaseAgent):

    def __init__(self, name="Greedy Chain Bot"):
        super().__init__(name)


    def get_move(self, s):

        moves = s.get_valid_moves()

        best_score = -10**9
        best_moves = []


        for move in moves:

            score = self.evaluate_move(s, move)

            if score > best_score:
                best_score = score
                best_moves = [move]

            elif score == best_score:
                best_moves.append(move)


        return random.choice(best_moves)



    def evaluate_move(self, s, move):

        score = 0


        before = self.count_boxes(s)


        s.execute_move(move)

        try:

            after = self.count_boxes(s)

            captured = after - before

            # 1. Reward capture
            score += captured * 10000


            # 2. Count boxes opponent can take
            opponent_gain = 0

            for m in s.get_valid_moves():

                if self._creates_box(s,m):
                    opponent_gain += 1


            score -= opponent_gain * 1000


            # 3. Chain penalty
            chain_count = self.count_chains(s)

            score -= chain_count * 100


            # 4. Safe position
            if opponent_gain == 0:
                score += 50


        finally:
            s.undo_move()


        return score



    def count_boxes(self,s):

        return sum(
            1 for x in s.b.flatten()
            if x != 0
        )


    def count_chains(self,s):

        """
        Count boxes with 2 or 3 edges.
        More 3-edge boxes means dangerous chain.
        """

        count = 0

        for r in range(s.SIZE):
            for c in range(s.SIZE):

                if s.b[r][c] == 0:

                    lines = s.get_lines_of_box((r,c))

                    filled = sum(
                        1 for ln in lines
                        if s.l[ln] != 0
                    )

                    if filled >= 2:
                        count += 1

        return count



    def _creates_box(self,s,move):

        for box in s.get_boxes_of_line(move):

            lines=s.get_lines_of_box(box)

            if sum(
                1 for ln in lines
                if s.l[ln]!=0
            ) == 3:
                return True

        return False