import os
import argparse
import multiprocessing as mp
import time
from tqdm import tqdm
from tournament import create_agent, RandomAgent

def run_single_game_speed(args):
    agent_name, size, game_idx = args
    from game import DotsAndBoxesGame
    
    agent = create_agent(agent_name, size)
    random_agent = RandomAgent()
    
    # Alternate who starts to ensure fairness and varied game trees
    starting_player = 1 if (game_idx % 2 == 0) else -1
    game = DotsAndBoxesGame(size=size, starting_player=starting_player, early_stopping=True)
    
    agent_moves = 0
    agent_time = 0.0
    
    while game.is_running():
        if game.current_player == starting_player:
            start_t = time.time()
            move = agent.get_move(game)
            agent_time += (time.time() - start_t)
            agent_moves += 1
        else:
            move = random_agent.get_move(game)
            
        game.execute_move(move)
        
    return agent_moves, agent_time

def main():
    parser = argparse.ArgumentParser(description="Measure agent decision speed vs Random Agent.")
    parser.add_argument("--size", type=int, default=5, help="Grid size of the Dots and Boxes board (default: 5).")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play per agent (default: 10).")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes.")
    args = parser.parse_args()

    # List of agent names to check speed
    agent_names = [
        # "Random", 
         "Greedy", 
         "Greedy Chain",
        # "Alpha-Beta (0.1s)", 
        # "Alpha-Beta (0.5s)", 
        # "Alpha-Beta v1 (0.1s)",
        # "Alpha-Beta v1 (0.5s)",
        # "Alpha-Beta v2 (0.1s)",
        # "Alpha-Beta v2 (0.5s)",
        # "MCTS (0.1s)", 
        # "MCTS (0.5s)",
        # "MCTS (100sims)",
        # "MCTS (200sims)",
        # "UCLABot",
        # "UCLABot_v2",
        "UCLABot_v3",
         "UCLABot_v4", 
         "UCLABot_v5", 
         "UCLABot_v6",
        # "UCLAGreedyBot",
        # "UCLAAlphaBeta",
        # "UCLA_MCTS_100",
        # # "UCLA_MCTS_200",
        #  "SimpleBot",
        #  "SimpleBot_v2",
        # "UCLA_MCTS_0.1",
        # "UCLA_MCTS_0.2",
    ]
    num_cores = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
    
    print(f"Starting Speed Test vs Random Agent")
    print(f"Grid Size: {args.size}x{args.size}")
    print(f"Games per Agent: {args.games}")
    print(f"Workers: {num_cores}")
    print("-" * 70)
    
    speed_results = []
    
    for agent_name in agent_names:
        print(f"\nEvaluating speed for: {agent_name}")
        tasks = [(agent_name, args.size, i) for i in range(args.games)]
        
        total_moves = 0
        total_time = 0.0
        
        try:
            with mp.Pool(num_cores) as pool:
                pbar = tqdm(pool.imap_unordered(run_single_game_speed, tasks), total=len(tasks), desc=f"{agent_name}")
                for moves, elapsed in pbar:
                    total_moves += moves
                    total_time += elapsed
            
            mps = total_moves / total_time if total_time > 0 else 0.0
            ms_per_move = (total_time / total_moves) * 1000 if total_moves > 0 else 0.0
            
            speed_results.append({
                "name": agent_name,
                "moves": total_moves,
                "time": total_time,
                "mps": mps,
                "ms_per_move": ms_per_move
            })
        except Exception as e:
            print(f"Error testing agent {agent_name}: {e}")
            
    # Sort results by moves/sec descending (fastest first)
    speed_results.sort(key=lambda x: x["mps"], reverse=True)
    
    print("\n" + "=" * 80)
    print(f"{'Agent Name':<25} | {'Moves':<8} | {'Total Time (s)':<15} | {'Moves/Sec':<12} | {'Avg ms/Move':<12}")
    print("=" * 80)
    for res in speed_results:
        print(f"{res['name']:<25} | {res['moves']:<8} | {res['time']:<15.3f} | {res['mps']:<12.2f} | {res['ms_per_move']:<12.2f}")
    print("=" * 80)

if __name__ == "__main__":
    mp.freeze_support()
    main()
