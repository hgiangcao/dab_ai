import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from game import DotsAndBoxesGame

def analyze_game_phase(filepath: str = "game_logs_bot.jsonl"):
    if not os.path.exists(filepath):
        print(f"Error: Log file '{filepath}' not found.")
        return

    print(f"Reading and analyzing game logs from {filepath}...")
    
    # Store scores for each move step: move_num -> list of scores
    p1_scores_by_move = {}
    p2_scores_by_move = {}
    
    # Store scores by total game move count
    p1_scores_by_total_move = {}
    p2_scores_by_total_move = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Processing games"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
            
        moves = record.get("moves", [])
        winner = record.get("winner", 0)
        if not moves:
            continue
            
        # Infer size from moves
        max_move = max(moves)
        size = 5
        for s in range(1, 20):
            if 2 * s * (s + 1) > max_move:
                size = s
                break
                
        # First pass to determine the correct starting player
        test_game = DotsAndBoxesGame(size=size, starting_player=1)
        for move in moves:
            if move in test_game.get_valid_moves():
                test_game.execute_move(move)
            else:
                break
                
        if test_game.result == winner:
            correct_starting_player = 1
        elif test_game.result == -winner:
            correct_starting_player = -1
        else:
            correct_starting_player = 1
            
        # Replay game with correct starting player and collect scores
        game = DotsAndBoxesGame(size=size, starting_player=correct_starting_player)
        
        p1_moves = 0
        p2_moves = 0
        total_moves = 0
        
        for move in moves:
            active_player = game.current_player
            game.execute_move(move)
            total_moves += 1
            
            p1_score = int(np.sum(game.b == 1))
            p2_score = int(np.sum(game.b == -1))
            
            if total_moves not in p1_scores_by_total_move:
                p1_scores_by_total_move[total_moves] = []
                p2_scores_by_total_move[total_moves] = []
            
            p1_scores_by_total_move[total_moves].append(p1_score)
            p2_scores_by_total_move[total_moves].append(p2_score)
            
            if active_player == 1:
                p1_moves += 1
                if p1_moves not in p1_scores_by_move:
                    p1_scores_by_move[p1_moves] = []
                p1_scores_by_move[p1_moves].append(p1_score)
            else:
                p2_moves += 1
                if p2_moves not in p2_scores_by_move:
                    p2_scores_by_move[p2_moves] = []
                p2_scores_by_move[p2_moves].append(p2_score)

    # Calculate average and standard deviation
    p1_steps = sorted(p1_scores_by_move.keys())
    p2_steps = sorted(p2_scores_by_move.keys())
    
    p1_avg = [np.mean(p1_scores_by_move[i]) for i in p1_steps]
    p1_std = [np.std(p1_scores_by_move[i]) for i in p1_steps]
    
    p2_avg = [np.mean(p2_scores_by_move[i]) for i in p2_steps]
    p2_std = [np.std(p2_scores_by_move[i]) for i in p2_steps]
    
    t_steps = sorted(p1_scores_by_total_move.keys())
    p1_t_avg = [np.mean(p1_scores_by_total_move[i]) for i in t_steps]
    p1_t_std = [np.std(p1_scores_by_total_move[i]) for i in t_steps]
    
    p2_t_avg = [np.mean(p2_scores_by_total_move[i]) for i in t_steps]
    p2_t_std = [np.std(p2_scores_by_total_move[i]) for i in t_steps]

    # Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Player 1 and Player 2 curves and std shading (Independent Move Count)
    ax1.plot(p1_steps, p1_avg, color='#1f77b4', linewidth=2.5, label='Player 1 (Avg)')
    ax1.fill_between(p1_steps, np.array(p1_avg) - np.array(p1_std), np.array(p1_avg) + np.array(p1_std), 
                     color='#1f77b4', alpha=0.15)
                     
    ax1.plot(p2_steps, p2_avg, color='#ff7f0e', linewidth=2.5, label='Player 2 (Avg)')
    ax1.fill_between(p2_steps, np.array(p2_avg) - np.array(p2_std), np.array(p2_avg) + np.array(p2_std), 
                     color='#ff7f0e', alpha=0.15)
                     
    ax1.set_title("Average Score vs Independent Move Count", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Independent Move Count (up to ~30)", fontsize=12)
    ax1.set_ylabel("Score (Boxes Captured)", fontsize=12)
    ax1.legend(loc='upper left')
    
    # 2. Player 1 and Player 2 curves and std shading (Total Game Moves)
    ax2.plot(t_steps, p1_t_avg, color='#1f77b4', linewidth=2.5, label='Player 1 (Avg)')
    ax2.fill_between(t_steps, np.array(p1_t_avg) - np.array(p1_t_std), np.array(p1_t_avg) + np.array(p1_t_std), 
                     color='#1f77b4', alpha=0.15)
                     
    ax2.plot(t_steps, p2_t_avg, color='#ff7f0e', linewidth=2.5, label='Player 2 (Avg)')
    ax2.fill_between(t_steps, np.array(p2_t_avg) - np.array(p2_t_std), np.array(p2_t_avg) + np.array(p2_t_std), 
                     color='#ff7f0e', alpha=0.15)
                     
    ax2.set_title("Average Score vs Total Game Moves", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Total Game Moves (up to 60)", fontsize=12)
    ax2.set_ylabel("Score (Boxes Captured)", fontsize=12)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    output_img = "game_phase_analysis.png"
    plt.savefig(output_img, dpi=300)
    print(f"Visualization saved to {output_img}")

if __name__ == "__main__":
    analyze_game_phase()
