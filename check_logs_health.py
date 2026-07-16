import os
import json
import numpy as np
from game import DotsAndBoxesGame

def check_logs_health(filepath: str = "game_logs_bot.jsonl"):
    if not os.path.exists(filepath):
        print(f"Error: Log file '{filepath}' not found. Please run log generation first!")
        return

    game_lengths = []
    score_margins = []
    first_box_turns = []
    total_games = 0
    p1_wins = 0
    p2_wins = 0
    draws = 0
    heatmaps_by_size = {}

    print(f"Analyzing log file: {filepath} ...")

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except Exception as e:
                print(f"Skipping corrupt line: {e}")
                continue
            
            moves = record.get("moves", [])
            if not moves:
                continue
            
            total_games += 1
            game_lengths.append(len(moves))
            
            # Recreate game to track scores and first box claim turn
            # Infer board size
            max_move = max(moves)
            size = 5
            for s in range(1, 20):
                if 2 * s * (s + 1) > max_move:
                    size = s
                    break
                    
            game = DotsAndBoxesGame(size=size, starting_player=1)
            first_box_turn = None
            
            for turn_idx, move in enumerate(moves):
                game.execute_move(move)
                
                # Check if a box has been claimed
                if first_box_turn is None and np.sum(game.b != 0) > 0:
                    first_box_turn = turn_idx + 1 # 1-indexed turn number

            # Final scores
            p1_score = int(np.sum(game.b == 1))
            p2_score = int(np.sum(game.b == -1))
            margin = abs(p1_score - p2_score)
            
            score_margins.append(margin)
            if first_box_turn is not None:
                first_box_turns.append(first_box_turn)

            # Record winner
            if game.result == 1:
                p1_wins += 1
            elif game.result == -1:
                p2_wins += 1
            else:
                draws += 1

            # Accumulate heatmap
            if size not in heatmaps_by_size:
                heatmaps_by_size[size] = (np.zeros((size + 1, size), dtype=int), np.zeros((size, size + 1), dtype=int))
            h_heatmap, v_heatmap = heatmaps_by_size[size]
            
            for move in moves[:10]:
                h_dummy, v_dummy = np.zeros(game.N_LINES), np.zeros(game.N_LINES)
                h_dummy[move] = 1
                h, v = game.l_to_h_v(h_dummy)
                if np.any(h == 1):
                    r, c = np.where(h == 1)[0][0], np.where(h == 1)[1][0]
                    h_heatmap[r][c] += 1
                else:
                    h_dummy[move] = 0; v_dummy[move] = 1
                    _, v = game.l_to_h_v(v_dummy)
                    r, c = np.where(v == 1)[0][0], np.where(v == 1)[1][0]
                    v_heatmap[r][c] += 1

    if total_games == 0:
        print("No valid game logs found in the file.")
        return

    # Helper to print simple text-based histogram/distribution
    def print_distribution(data, bins=5):
        counts, bin_edges = np.histogram(data, bins=bins)
        max_count = max(counts) if len(counts) > 0 else 1
        for i in range(len(counts)):
            bar = "#" * int((counts[i] / max_count) * 30)
            print(f"  [{bin_edges[i]:5.1f} - {bin_edges[i+1]:5.1f}]: {counts[i]:5d} {bar}")

    print("\n" + "="*50)
    print(f"LOG HEALTH REPORT ({total_games} Games)")
    print("="*50)

    # 1. Game Length Distribution
    print(f"\n1. Game Length Distribution (Total moves taken per game):")
    print(f"  Min: {np.min(game_lengths)}")
    print(f"  Max: {np.max(game_lengths)}")
    print(f"  Mean: {np.mean(game_lengths):.2f}")
    print(f"  Median: {np.median(game_lengths):.1f}")
    print_distribution(game_lengths)

    # 2. Final Score Margin
    print(f"\n2. Final Score Margin (abs(Player 1 Score - Player 2 Score)):")
    print(f"  Min: {np.min(score_margins)}")
    print(f"  Max: {np.max(score_margins)}")
    print(f"  Mean: {np.mean(score_margins):.2f}")
    print(f"  Median: {np.median(score_margins):.1f}")
    print_distribution(score_margins)

    # 3. First Box Claimed (Turn Number)
    if first_box_turns:
        print(f"\n3. First Box Claimed (Turn/Ply index when first box was completed):")
        print(f"  Min Turn: {np.min(first_box_turns)}")
        print(f"  Max Turn: {np.max(first_box_turns)}")
        print(f"  Mean Turn: {np.mean(first_box_turns):.2f}")
        print(f"  Median Turn: {np.median(first_box_turns):.1f}")
        print_distribution(first_box_turns)
    else:
        print(f"\n3. First Box Claimed: No boxes were claimed in any game.")

    # 4. Win/Loss/Draw Ratio
    print(f"\n4. Win/Loss/Draw Ratio (Player 1 Advantage check):")
    p1_pct = (p1_wins / total_games) * 100
    p2_pct = (p2_wins / total_games) * 100
    draw_pct = (draws / total_games) * 100
    print(f"  Player 1 (Red) Wins  : {p1_wins:5d} ({p1_pct:5.2f}%)")
    print(f"  Player 2 (Blue) Wins : {p2_wins:5d} ({p2_pct:5.2f}%)")
    print(f"  Draws                : {draws:5d} ({draw_pct:5.2f}%)")

    # 5. Action Heatmap
    print(f"\n5. Action Heatmap (Selection count per edge):")
    for size, (h_heatmap, v_heatmap) in heatmaps_by_size.items():
        print(f"\n  Board Size: {size}x{size}")
        print("  Horizontal Edges:")
        for r in range(size + 1):
            row_str = " ".join(f"{val:6d}" for val in h_heatmap[r])
            print(f"    Row {r}: {row_str}")
        print("  Vertical Edges:")
        for r in range(size):
            row_str = " ".join(f"{val:6d}" for val in v_heatmap[r])
            print(f"    Row {r}: {row_str}")
    print("="*50 + "\n")

if __name__ == "__main__":
    check_logs_health()
