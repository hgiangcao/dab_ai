import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from game import DotsAndBoxesGame
from bots.alphazero_agent import AlphaZeroAgent

def main():
    sim_counts = [0, 100, 200, 300, 500, 1000]
    
    # Initialize an empty 5x5 game
    game = DotsAndBoxesGame(size=5)
    
    print(f"--- Debugging AlphaZero First Move Probabilities ---")
    print(f"Board Size: {game.SIZE}x{game.SIZE}")
    print(f"Total Lines: {game.N_LINES}\n")

    for n_sim in sim_counts:
        print(f"=== {n_sim} Simulations ===")
        # Note: model_path default is "best.pth.tar" in AlphaZeroAgent
        agent = AlphaZeroAgent(name=f"AZ_{n_sim}", n_simulations=n_sim)
        agent.reset()
        
        # Get probabilities with temp=1 to see the distribution (visit counts or raw policy)
        pi = agent.mcts.play(game, temp=1)
        
        # Filter out zero probabilities and sort by probability (descending)
        move_probs = [(move, prob) for move, prob in enumerate(pi) if prob > 1e-4]
        move_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Print top moves
        print(f"Moves (Move Number -> Probability | Visits):")
        for move, prob in move_probs:
            visits = agent.mcts._root.N.get(move, 0) if agent.mcts._root is not None else 0
            print(f"  Move {move:2d} : {prob:.4f} | Visits: {visits}")
        print()

if __name__ == "__main__":
    main()
