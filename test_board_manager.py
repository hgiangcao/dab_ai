"""
test_board_manager.py
=====================
Comprehensive consistency test between the game engine (game.py) and the
Board_Manager (game_board_manager.py).

Checks performed after every move and every undo:
  1. Edge occupancy:     bm_edge.occupied  <-> game.l[line_idx] != 0
  2. Point degree:       bm_point.degree   <-> number of occupied edges at that point
  3. Box edge_count:     bm_box.edge_count <-> game.box_fill_count[box_idx]
  4. Box owner:          bm_box.owner      <-> game.b[row][col]  (0->None, else 1 or -1)
  5. box_by_edge_count:  the 5 sets are mutually exclusive, cover all boxes, and
                         match each box's own edge_count field.
  6. 0-1-2-3-4 edge bucket query printed at each step.
"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import DotsAndBoxesGame

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}OK  {msg}{RESET}")
def fail(msg):
    print(f"  {RED}FAIL {msg}{RESET}")
    raise AssertionError(msg)


def check_consistency(game: DotsAndBoxesGame, label: str = ""):
    bm   = game.board_manager
    SIZE = game.SIZE

    # 1. Edge occupancy
    for line_idx, bm_edge in game._line_to_bm_edge.items():
        game_occupied = game.l[line_idx] != 0
        if bm_edge.occupied != game_occupied:
            fail(f"{label} | Edge {line_idx}: bm.occupied={bm_edge.occupied}  game.l={game.l[line_idx]}")

    # 2. Point degree
    for row_pts in bm.points:
        for pt in row_pts:
            expected_degree = sum(1 for e in pt.edges if e.occupied)
            if pt.degree != expected_degree:
                fail(f"{label} | Point ({pt.row},{pt.col}): bm.degree={pt.degree}  expected={expected_degree}")

    # 3. Box edge_count & owner
    for box_idx, bm_box in enumerate(bm.boxes):
        row, col = divmod(box_idx, SIZE)

        expected_ec = game.box_fill_count[box_idx]
        if bm_box.edge_count != expected_ec:
            fail(f"{label} | Box {box_idx}: bm.edge_count={bm_box.edge_count}  game.fill={expected_ec}")

        game_owner_raw = int(game.b[row][col])
        expected_owner = game_owner_raw if game_owner_raw != 0 else None
        if bm_box.owner != expected_owner:
            fail(f"{label} | Box {box_idx}: bm.owner={bm_box.owner}  game.b={game_owner_raw}")

    # 4. box_by_edge_count sets
    seen = set()
    for bucket_n, bucket in enumerate(bm.box_by_edge_count):
        for box_id in bucket:
            if box_id in seen:
                fail(f"{label} | Box {box_id} appears in multiple buckets!")
            seen.add(box_id)
            if bm.boxes[box_id].edge_count != bucket_n:
                fail(f"{label} | Box {box_id} in bucket {bucket_n} but edge_count={bm.boxes[box_id].edge_count}")
    if len(seen) != game.N_BOXES:
        fail(f"{label} | bucket union covers {len(seen)} boxes, expected {game.N_BOXES}")


def print_buckets(game: DotsAndBoxesGame, label: str = ""):
    bm = game.board_manager
    parts = [f"{n}-edge:{len(bm.box_by_edge_count[n])}" for n in range(5)]
    print(f"  {YELLOW}[{label}] {' | '.join(parts)}{RESET}")


def test_random_game(size=5, seed=42):
    print(f"\n{'='*55}")
    print(f" TEST random game  size={size}  seed={seed}")
    print(f"{'='*55}")

    random.seed(seed)
    game = DotsAndBoxesGame(size=size, starting_player=1)

    check_consistency(game, "initial")
    print_buckets(game, "start")

    move_count = 0
    while game.is_running():
        line = random.choice(game.get_valid_moves())
        game.execute_move(line)
        move_count += 1
        check_consistency(game, f"move {move_count} line={line}")
        if move_count % 10 == 0:
            print_buckets(game, f"move {move_count}")

    print_buckets(game, "game over")
    ok(f"Full game: {move_count} moves all consistent")

    print(f"  Undoing all {move_count} moves...")
    for step in range(move_count, 0, -1):
        game.undo_move()
        check_consistency(game, f"undo {step}")

    print_buckets(game, "after full undo")
    assert len(game.board_manager.box_by_edge_count[0]) == game.N_BOXES
    ok(f"All {move_count} undos verified correctly")


def test_interleaved(size=5, seed=99):
    print(f"\n{'='*55}")
    print(f" TEST interleaved execute/undo  size={size}  seed={seed}")
    print(f"{'='*55}")

    random.seed(seed)
    game = DotsAndBoxesGame(size=size, starting_player=1)
    ops = 0

    for _ in range(200):
        valid = game.get_valid_moves()
        if not valid:
            break
        if game._history and random.random() < 0.30:
            game.undo_move()
        else:
            game.execute_move(random.choice(valid))
        ops += 1
        check_consistency(game, f"op {ops}")

    ok(f"Interleaved: {ops} operations all consistent")


def test_clone(size=5, seed=7):
    print(f"\n{'='*55}")
    print(f" TEST clone isolation  size={size}  seed={seed}")
    print(f"{'='*55}")

    random.seed(seed)
    game = DotsAndBoxesGame(size=size, starting_player=1)

    for _ in range(20):
        valid = game.get_valid_moves()
        if not valid: break
        game.execute_move(random.choice(valid))

    clone = game.clone()
    check_consistency(game,  "original after 20 moves")
    check_consistency(clone, "clone after 20 moves")

    for _ in range(10):
        valid = clone.get_valid_moves()
        if not valid: break
        clone.execute_move(random.choice(valid))

    check_consistency(game,  "original (must be unchanged)")
    check_consistency(clone, "clone after 30 moves")
    ok("Clone is independent and consistent")


def test_bucket_queries(size=3, seed=0):
    print(f"\n{'='*55}")
    print(f" TEST bucket queries  size={size}  seed={seed}")
    print(f"{'='*55}")

    random.seed(seed)
    game = DotsAndBoxesGame(size=size, starting_player=1)
    bm   = game.board_manager
    moves = 0

    while game.is_running():
        line = random.choice(game.get_valid_moves())
        game.execute_move(line)
        moves += 1

        print(f"\n  After move {moves} (line={line}):")
        for n in range(5):
            ids = sorted(bm.box_by_edge_count[n])
            print(f"    {n}-edge boxes (ids): {ids}")

        # Cross-check bucket vs game.box_fill_count
        for n in range(5):
            expected = sorted(i for i, cnt in enumerate(game.box_fill_count) if cnt == n)
            actual   = sorted(bm.box_by_edge_count[n])
            if expected != actual:
                fail(f"Bucket {n} mismatch: bm={actual}  expected={expected}")

    ok(f"Bucket queries verified over {moves} moves")


if __name__ == "__main__":
    try:
        test_random_game(size=5, seed=42)
        test_random_game(size=3, seed=13)
        test_interleaved(size=5, seed=99)
        test_clone(size=5, seed=7)
        test_bucket_queries(size=3, seed=0)

        print(f"\n{GREEN}{'='*55}")
        print(f"  ALL TESTS PASSED")
        print(f"{'='*55}{RESET}\n")

    except AssertionError as e:
        print(f"\n{RED}FAILED: {e}{RESET}\n")
        sys.exit(1)
