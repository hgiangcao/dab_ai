"""
game_gui_check_policy.py
────────────────────────
Visual policy inspector for the AlphaZero network on Dots and Boxes.

Features
--------
• Dropdown: opponent bot  (Greedy, UCLABot_v6, Random)
• "Play Game" button: auto-plays a full game; records every move + NN policy
• "◀ Prev" / "▶ Next" buttons: traverse history move-by-move
• Board: draws all lines + filled boxes (player colors)
• Policy heatmap: when the current move was chosen by AlphaZero (0 sims),
  each of the 60 possible line positions is colorized red→blue
  proportional to the raw NN probability (hot = high probability).
  The chosen move is circled in gold. Invalid moves are gray.

Usage
-----
    python game_gui_check_policy.py
    python game_gui_check_policy.py --model best.pth.tar
"""

import os, sys, argparse, math, colorsys, threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# ── project root on path ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from game import DotsAndBoxesGame
import config as cfg


# ── colour helpers ────────────────────────────────────────────────────────────

def prob_to_hex(p: float, p_min: float = 0.0, p_max: float = 1.0) -> str:
    """Map probability p → hex colour  (cold blue → hot red)."""
    if p_max <= p_min:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (p - p_min) / (p_max - p_min)))
    # Hue: 240° (blue) → 0° (red) as t goes 0→1
    hue = (1.0 - t) * 0.667
    r, g, b = colorsys.hsv_to_rgb(hue, 0.90, 0.95)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


# ── agent factory ─────────────────────────────────────────────────────────────

def make_bot(name: str):
    """Return an agent object by name string."""
    if name == "Greedy":
        from bots.greedy import GreedyPlayer
        return GreedyPlayer(name=name)
    elif name == "UCLABot_v6":
        from bots.ucla_bot_v6 import UCLABot_v6
        return UCLABot_v6(name=name)
    elif name == "UCLABot_v3":
        from bots.ucla_bot import UCLABot_v3
        return UCLABot_v3(name=name)
    elif name == "Random":
        import random
        from agent_interface import BaseAgent
        class _Rnd(BaseAgent):
            def get_move(self, g):
                ms = g.get_valid_moves()
                return random.choice(ms) if ms else None
        return _Rnd(name="Random")
    else:
        raise ValueError(f"Unknown bot: {name}")


def make_az_agent(model_path: str, n_sims: int = 100):
    """Load AlphaZero nnet from checkpoint; return (nnet, MCTS) objects."""
    import torch
    from model import NNetWrapper, dotdict
    from mcts import MCTS

    dummy = DotsAndBoxesGame(size=5, starting_player=1)
    args = dotdict({
        "lr": 1e-3, "epochs": 1, "batch_size": 512,
        "num_channels": 256, "num_res_blocks": 10,
        "l2_reg": 1e-4, "lr_scheduler_steps": 336,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    })
    nnet = NNetWrapper(dummy, args)
    dev  = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=dev, weights_only=False)
        weights = state.get("state_dict", state)
        nnet.nnet.load_state_dict(weights, strict=True)
        print(f"[AZ] Loaded weights from {model_path}")
    else:
        print(f"[AZ] WARNING: checkpoint not found ({model_path}), using random weights")
    nnet.nnet.eval()

    mcts_params = {
        "n_simulations":   n_sims,
        "c_puct":          cfg.MCTS_C_PUCT,
        "dirichlet_eps":   0.0,
        "dirichlet_alpha": cfg.MCTS_DIRICHLET_ALPHA,
    }
    mcts = MCTS(nnet, mcts_params)
    return nnet, mcts


def encode_state(game: DotsAndBoxesGame, nnet) -> np.ndarray:
    """Build the (4, S+1, S+1) board tensor the way MCTS does."""
    size = game.SIZE
    sp1  = size + 1
    buf  = np.zeros((4, sp1, sp1), dtype=np.float32)
    canonical_lines = game.get_canonical_lines()
    h, v = game.l_to_h_v(canonical_lines)
    buf[0, :sp1, :size] = h
    buf[1, :size, :sp1] = v
    cboxes = game.get_canonical_boxes()
    buf[2, :size, :size] = np.where(cboxes ==  1, 1.0, 0.0)
    buf[3, :size, :size] = np.where(cboxes == -1, 1.0, 0.0)
    return buf


def get_subtree_depth(node) -> int:
    """Recursively calculate the maximum depth explored in a node's subtree."""
    if not node.children:
        return 0
    return 1 + max(get_subtree_depth(c) for c in node.children.values())


# ── record of one move step ───────────────────────────────────────────────────

class MoveRecord:
    """Everything we want to display for one step in the game."""
    __slots__ = ("move_idx", "move", "player", "agent_label", "policy", "value",
                 "lines_snapshot", "boxes_snapshot", "explored_count", "valid_count", "visit_counts", "child_depths")

    def __init__(self, move_idx, move, player, agent_label,
                 policy, value, lines_snapshot, boxes_snapshot,
                 explored_count=None, valid_count=None, visit_counts=None, child_depths=None):
        self.move_idx       = move_idx        # 0-based
        self.move           = move            # line index
        self.player         = player          # 1 or -1
        self.agent_label    = agent_label     # "AlphaZero" or bot name
        self.policy         = policy          # np.ndarray[60] or None
        self.value          = value           # float or None
        self.lines_snapshot = lines_snapshot  # copy of game.l BEFORE move
        self.boxes_snapshot = boxes_snapshot  # copy of game.b BEFORE move
        self.explored_count = explored_count
        self.valid_count    = valid_count
        self.visit_counts   = visit_counts
        self.child_depths   = child_depths


# ─────────────────────────────────────────────────────────────────────────────
# Main GUI class
# ─────────────────────────────────────────────────────────────────────────────

class PolicyInspectorGUI:
    SIZE    = 5
    SPACING = 72
    OFFSET  = 52
    DOT_R   = 7
    LINE_W  = 6
    HIT_W   = 12   # hit-test half-width for line detection

    def __init__(self, root: tk.Tk, model_path: str):
        self.root       = root
        self.model_path = model_path
        self.root.title("AlphaZero Policy Inspector — Dots & Boxes")
        self.root.resizable(True, True)

        # game state
        self._history: list[MoveRecord] = []
        self._step    = 0    # which step we're viewing (0 = blank start)
        self._playing = False

        # AZ network (loaded once)
        self._nnet = None
        self._mcts = None

        self._build_ui()
        self._draw_board()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── top control bar ──
        bar = tk.Frame(self.root, bg="#f5f5f5", pady=8)
        bar.pack(fill=tk.X)

        tk.Label(bar, text="AlphaZero Policy Inspector",
                 font=("Segoe UI", 13, "bold"),
                 bg="#f5f5f5", fg="#1a1a2e").pack(side=tk.LEFT, padx=14)

        # opponent selector
        tk.Label(bar, text="Opponent:", bg="#f5f5f5", fg="#1a7a1a",
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(20, 4))
        self._opp_var = tk.StringVar(value="UCLABot_v6")
        self._opp_combo = ttk.Combobox(
            bar, textvariable=self._opp_var,
            values= ["UCLABot_v6", "UCLABot_v3", "Random"],
            state="readonly", width=14)
        self._opp_combo.pack(side=tk.LEFT)

        # AlphaZero plays as…
        tk.Label(bar, text="AZ plays as:", bg="#f5f5f5", fg="#1a7a1a",
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(16, 4))
        self._az_player_var = tk.StringVar(value="Player 1")
        ttk.Combobox(
            bar, textvariable=self._az_player_var,
            values=["Player 1", "Player 2"],
            state="readonly", width=10).pack(side=tk.LEFT)

        # MCTS Sims
        tk.Label(bar, text="Sims:", bg="#f5f5f5", fg="#1a7a1a",
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(16, 4))
        self._sims_var = tk.StringVar(value="100")
        ttk.Combobox(
            bar, textvariable=self._sims_var,
            values=["0", "50", "100", "200", "500", "1000", "2000"],
            state="readonly", width=6).pack(side=tk.LEFT)

        # Play button
        self._btn_play = tk.Button(
            bar, text="▶  Play Game",
            font=("Segoe UI", 10, "bold"),
            bg="#1a7a1a", fg="#f5f5f5",
            activebackground="#4a9a4a",
            relief=tk.FLAT, padx=10, pady=4,
            command=self._on_play)
        self._btn_play.pack(side=tk.LEFT, padx=18)

        # ── status line ──
        self._status_var = tk.StringVar(value="Press ▶ Play Game to start.")
        status_bar = tk.Frame(self.root, bg="#eeeeee")
        status_bar.pack(fill=tk.X)
        tk.Label(status_bar, textvariable=self._status_var,
                 font=("Segoe UI", 10), bg="#eeeeee", fg="#1a1a2e",
                 anchor="w", padx=10, pady=4).pack(fill=tk.X)

        # ── main area: board left | separator | policy panel right ──
        # Use a PanedWindow so both columns are always visible
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                              bg="#cccccc", sashwidth=4,
                              sashrelief=tk.FLAT, bd=0)
        pane.pack(fill=tk.BOTH, expand=True)

        # --- LEFT pane: game board ---
        left_pane = tk.Frame(pane, bg="#f5f5f5")
        pane.add(left_pane, minsize=300)

        board_size = 2 * self.OFFSET + self.SIZE * self.SPACING
        self._canvas = tk.Canvas(
            left_pane, width=board_size, height=board_size,
            bg="#f5f5f5", highlightthickness=0)
        self._canvas.pack(padx=16, pady=16)

        # --- RIGHT pane: policy panel ---
        self._policy_frame = tk.Frame(pane, bg="#eeeeee")
        pane.add(self._policy_frame, minsize=360)
        # Set initial sash position after window is shown
        self.root.after(50, lambda: pane.sash_place(0, board_size + 32, 0))

        # policy canvas (same layout grid as the board, scaled down)
        pol_size = 2 * 30 + self.SIZE * 54
        self._pol_canvas = tk.Canvas(
            self._policy_frame, width=pol_size, height=pol_size,
            bg="#eeeeee", highlightthickness=0)
        self._pol_canvas.pack(pady=(12, 4))

        self._pol_title = tk.Label(
            self._policy_frame, text="Raw NN Policy",
            font=("Segoe UI", 11, "bold"), bg="#eeeeee", fg="#1a1a2e")
        self._pol_title.pack()

        self._val_label = tk.Label(
            self._policy_frame, text="",
            font=("Segoe UI", 10), bg="#eeeeee", fg="#1a7a1a")
        self._val_label.pack(pady=2)

        # legend
        leg = tk.Frame(self._policy_frame, bg="#eeeeee")
        leg.pack(pady=6)
        for prob, lbl in [(1.0, "High"), (0.5, ""), (0.0, "Low")]:
            c = prob_to_hex(prob)
            f = tk.Frame(leg, bg=c, width=22, height=14)
            f.pack(side=tk.LEFT, padx=1)
            if lbl:
                tk.Label(leg, text=lbl, bg="#eeeeee", fg="#1a1a2e",
                         font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=2)

        # ── sorted probability list ──
        tk.Label(self._policy_frame,
                 text="Sorted Move Probabilities",
                 font=("Segoe UI", 9, "bold"),
                 bg="#eeeeee", fg="#1a1a2e").pack(pady=(4, 0))

        list_frame = tk.Frame(self._policy_frame, bg="#eeeeee")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 8))

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._prob_list = tk.Listbox(
            list_frame,
            font=("Consolas", 9),
            bg="#ffffff", fg="#1a1a2e",
            selectbackground="#d0e8ff",
            selectforeground="#1a1a2e",
            activestyle="none",
            yscrollcommand=scrollbar.set,
            height=15,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#cccccc")
        scrollbar.config(command=self._prob_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._prob_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── nav bar ──
        nav = tk.Frame(self.root, bg="#f5f5f5", pady=8)
        nav.pack(fill=tk.X)

        self._btn_prev = tk.Button(
            nav, text="◀  Prev",
            font=("Segoe UI", 10), bg="#dddddd", fg="#1a1a2e",
            activebackground="#cccccc", relief=tk.FLAT, padx=12, pady=4,
            command=self._prev_step)
        self._btn_prev.pack(side=tk.LEFT, padx=20)

        self._step_var = tk.StringVar(value="Step 0 / 0")
        tk.Label(nav, textvariable=self._step_var,
                 font=("Segoe UI", 10), bg="#f5f5f5", fg="#444444").pack(
                     side=tk.LEFT, expand=True)

        self._btn_next = tk.Button(
            nav, text="Next  ▶",
            font=("Segoe UI", 10), bg="#dddddd", fg="#1a1a2e",
            activebackground="#cccccc", relief=tk.FLAT, padx=12, pady=4,
            command=self._next_step)
        self._btn_next.pack(side=tk.RIGHT, padx=20)

    # ── game play ─────────────────────────────────────────────────────────────

    def _on_play(self):
        if self._playing:
            return
        self._playing = True
        self._btn_play.config(state=tk.DISABLED, text="Playing…")
        t = threading.Thread(target=self._run_game, daemon=True)
        t.start()

    def _run_game(self):
        try:
            self._do_run_game()
        except Exception as e:
            import traceback
            traceback.print_exc()
            err_msg = str(e)
            self.root.after(0, lambda msg=err_msg: messagebox.showerror("Error", msg))
        finally:
            self._playing = False
            self.root.after(0, lambda: self._btn_play.config(
                state=tk.NORMAL, text="▶  Play Game"))

    def _do_run_game(self):
        self.root.after(0, lambda: self._status_var.set("Loading AlphaZero model…"))

        # Load AZ model only once
        if self._nnet is None:
            self._nnet, self._mcts = make_az_agent(self.model_path, n_sims=0)

        opp_name   = self._opp_var.get()
        n_sims     = int(self._sims_var.get())
        az_is_p1   = self._az_player_var.get() == "Player 1"
        az_player  = 1 if az_is_p1 else -1
        opp_player = -az_player

        # Update MCTS sims
        self._mcts.n_simulations = n_sims

        self.root.after(0, lambda: self._status_var.set(
            f"Playing: AlphaZero ({n_sims} sims, P{'1' if az_is_p1 else '2'}) vs {opp_name}…"))

        opp_bot = make_bot(opp_name)
        self._mcts.reset_tree()
        game = DotsAndBoxesGame(size=self.SIZE, starting_player=1)

        history = []
        last_action = None

        while game.is_running():
            cur = game.current_player
            lines_snap = game.l.copy()
            boxes_snap = game.b.copy()

            if cur == az_player:
                if n_sims == 0:
                    # AlphaZero 0-sim: pure NN policy
                    board_enc = encode_state(game, self._nnet)
                    policy_raw, value = self._nnet.predict(board_enc)
    
                    valid = game.get_valid_moves()
                    masked = np.zeros(game.N_LINES, dtype=np.float64)
                    masked[valid] = policy_raw[valid]
                    s = masked.sum()
                    if s > 0:
                        masked /= s
    
                    move = int(np.argmax(masked))
                    pol   = masked
                    val   = float(value.flat[0]) if hasattr(value, 'flat') else float(value)
                    visit_counts = None
                    child_depths = None
                else:
                    # AlphaZero >0 sims: MCTS
                    # We call with temp=1 to get the actual search probabilities (visit counts distribution)
                    # for the GUI, but we still pick the single best move deterministically using argmax.
                    pol = self._mcts.play(game, temp=1, add_root_noise=False, last_action=last_action)
                    move = int(np.argmax(pol))
                    visit_counts = [self._mcts._root.N.get(a, 0) for a in range(game.N_LINES)]
                    
                    child_depths = {}
                    for a in range(game.N_LINES):
                        if a in self._mcts._root.children:
                            child_depths[a] = 1 + get_subtree_depth(self._mcts._root.children[a])
                        else:
                            child_depths[a] = 0
                    
                    # Just run the NN once to get the value for display
                    _, value = self._nnet.predict(encode_state(game, self._nnet))
                    val = float(value.flat[0]) if hasattr(value, 'flat') else float(value)

                label = f"AlphaZero ({n_sims} sims)"
                valid_count = len(game.get_valid_moves())
                explored_count = sum(1 for p in pol if p > 0)
            else:
                move  = opp_bot.get_move(game)
                label = opp_name
                pol   = None
                val   = None
                valid_count = None
                explored_count = None
                visit_counts = None
                child_depths = None

            last_action = move

            rec = MoveRecord(
                move_idx=len(history),
                move=move,
                player=cur,
                agent_label=label,
                policy=pol,
                value=val,
                lines_snapshot=lines_snap,
                boxes_snapshot=boxes_snap,
                explored_count=explored_count,
                valid_count=valid_count,
                visit_counts=visit_counts,
                child_depths=child_depths
            )
            history.append(rec)
            game.execute_move(move)

        # compute final score
        p1 = int(np.sum(game.b == 1))
        p2 = int(np.sum(game.b == -1))
        az_score  = p1 if az_is_p1 else p2
        opp_score = p2 if az_is_p1 else p1
        result_str = (
            f"Game over! AlphaZero {az_score}–{opp_score} {opp_name}  "
            f"({'Win' if az_score > opp_score else 'Loss' if az_score < opp_score else 'Draw'})"
        )

        # Store final board state as extra record (no move, for viewing end)
        history.append(MoveRecord(
            move_idx=len(history),
            move=-1, player=0, agent_label="END",
            policy=None, value=None,
            lines_snapshot=game.l.copy(),
            boxes_snapshot=game.b.copy(),
        ))

        self._history = history
        self._step    = len(history) - 1   # jump to end

        self.root.after(0, lambda: self._status_var.set(result_str))
        self.root.after(0, self._refresh_view)

    # ── navigation ────────────────────────────────────────────────────────────

    def _prev_step(self):
        if self._step > 0:
            self._step -= 1
            self._refresh_view()

    def _next_step(self):
        if self._step < len(self._history) - 1:
            self._step += 1
            self._refresh_view()

    def _refresh_view(self):
        total = len(self._history)
        self._step_var.set(f"Step {self._step} / {max(0, total - 1)}")
        self._draw_board()
        self._draw_policy()

    # ── board rendering ───────────────────────────────────────────────────────

    def _line_coords(self, line_idx: int):
        """Return (x1,y1,x2,y2,cx,cy) for a line given its flat index."""
        half = self.SIZE * (self.SIZE + 1)
        S = self.SIZE
        OFF = self.OFFSET
        SP  = self.SPACING
        if line_idx < half:          # horizontal
            r = line_idx // S
            c = line_idx  % S
            x1 = OFF + c * SP; y1 = OFF + r * SP
            x2 = x1 + SP;      y2 = y1
        else:                        # vertical
            idx = line_idx - half
            c   = idx // S
            r   = idx  % S
            x1 = OFF + c * SP; y1 = OFF + r * SP
            x2 = x1;           y2 = y1 + SP
        return x1, y1, x2, y2, (x1+x2)//2, (y1+y2)//2

    def _draw_board(self):
        cv = self._canvas
        cv.delete("all")
        S   = self.SIZE
        OFF = self.OFFSET
        SP  = self.SPACING
        DR  = self.DOT_R

        # --- background
        cv.config(bg="#f5f5f5")

        # get snapshot
        if not self._history:
            lines = np.zeros(60, dtype=np.float32)
            boxes = np.zeros((S, S), dtype=np.float32)
            last_move = -1
        else:
            rec   = self._history[self._step]
            lines = rec.lines_snapshot
            boxes = rec.boxes_snapshot
            last_move = rec.move if rec.move_idx == self._step else -1

        # box fills
        for r in range(S):
            for c in range(S):
                if boxes[r, c] != 0:
                    col = "#ff7675" if boxes[r, c] == 1 else "#74b9ff"
                    x1 = OFF + c * SP + DR
                    y1 = OFF + r * SP + DR
                    x2 = OFF + (c+1) * SP - DR
                    y2 = OFF + (r+1) * SP - DR
                    cv.create_rectangle(x1, y1, x2, y2, fill=col, outline="")

        # drawn lines
        h_mat, v_mat = DotsAndBoxesGame(size=S, starting_player=1).l_to_h_v(lines)
        for r in range(S+1):
            for c in range(S):
                if h_mat[r, c] != 0:
                    col = "#ff7675" if h_mat[r, c] == 1 else "#74b9ff"
                    x1 = OFF + c * SP; y1 = OFF + r * SP
                    cv.create_line(x1, y1, x1+SP, y1, fill=col, width=self.LINE_W,
                                   capstyle=tk.ROUND)
        for r in range(S):
            for c in range(S+1):
                if v_mat[r, c] != 0:
                    col = "#ff7675" if v_mat[r, c] == 1 else "#74b9ff"
                    x1 = OFF + c * SP; y1 = OFF + r * SP
                    cv.create_line(x1, y1, x1, y1+SP, fill=col, width=self.LINE_W,
                                   capstyle=tk.ROUND)

        # highlight last move
        if 0 <= last_move < 60:
            x1, y1, x2, y2, cx, cy = self._line_coords(last_move)
            cv.create_line(x1, y1, x2, y2, fill="#f9c74f", width=4,
                           dash=(6, 4), capstyle=tk.ROUND)

        # dots
        for i in range(S+1):
            for j in range(S+1):
                x = OFF + j * SP; y = OFF + i * SP
                cv.create_oval(x-DR, y-DR, x+DR, y+DR, fill="#1a1a2e", outline="")

        # step label on board
        if self._history and self._step < len(self._history):
            rec = self._history[self._step]
            lbl = (f"Move {rec.move_idx+1}  |  "
                   f"{'P1🔴' if rec.player==1 else 'P2🔵'}  "
                   f"{rec.agent_label}")
            if rec.agent_label == "END":
                lbl = "Game Over"
            cv.create_text(OFF + S*SP//2, 12, text=lbl,
                           font=("Segoe UI", 9, "bold"), fill="#1a1a2e")

    # ── policy heatmap ────────────────────────────────────────────────────────

    def _draw_policy(self):
        pc = self._pol_canvas
        pc.delete("all")

        S    = self.SIZE
        OFF  = 30
        SP   = 54
        DR   = 5
        LW   = 7

        if not self._history or self._step >= len(self._history):
            self._pol_title.config(text="Raw NN Policy  (no data)")
            self._val_label.config(text="")
            self._prob_list.delete(0, tk.END)
            return

        rec = self._history[self._step]

        if rec.policy is None:
            self._pol_title.config(
                text=f"Policy: {rec.agent_label}\n(no NN policy for this agent)")
            self._val_label.config(text="")
            self._prob_list.delete(0, tk.END)
            self._prob_list.insert(tk.END, f"  Move by {rec.agent_label} — no NN policy")
            # just draw blank board skeleton
            for i in range(S+1):
                for j in range(S+1):
                    x = OFF + j*SP; y = OFF + i*SP
                    pc.create_oval(x-DR, y-DR, x+DR, y+DR, fill="#aaaaaa", outline="")
            return

        policy  = np.array(rec.policy, dtype=float)  # force numpy array for MCTS lists
        chosen  = rec.move
        valid   = np.where(policy > 0)[0]
        p_min   = policy[valid].min() if len(valid) else 0.0
        p_max   = policy[valid].max() if len(valid) else 1.0

        pol_name = "Raw NN Policy (0 sims)" if "0 sims" in rec.agent_label else "MCTS Search Probs"
        self._pol_title.config(
            text=f"{pol_name}  —  {rec.agent_label}\n"
                 f"Move {rec.move_idx+1}  |  chosen line #{chosen}")
        v_txt = f"Value head: {rec.value:+.3f}" if rec.value is not None else ""
        if rec.explored_count is not None and rec.valid_count is not None:
            v_txt += f"   |   Explored: {rec.explored_count}/{rec.valid_count} moves"
        self._val_label.config(text=v_txt)

        # helper: line centre on policy canvas
        half = S * (S+1)

        def pol_coords(li):
            if li < half:
                r = li // S; c = li % S
                x1 = OFF+c*SP; y1 = OFF+r*SP; x2 = x1+SP; y2 = y1
            else:
                idx = li - half; col = idx//S; row = idx%S
                x1 = OFF+col*SP; y1 = OFF+row*SP; x2 = x1; y2 = y1+SP
            return x1, y1, x2, y2, (x1+x2)//2, (y1+y2)//2

        # Draw heatmap segments for every line
        for li in range(60):
            x1, y1, x2, y2, cx, cy = pol_coords(li)
            p = policy[li]
            if p <= 0:
                # invalid / zero probability — draw grey
                col = "#313244"
                pc.create_line(x1, y1, x2, y2, fill=col, width=3,
                               capstyle=tk.ROUND)
            else:
                col = prob_to_hex(p, p_min, p_max)
                pc.create_line(x1, y1, x2, y2, fill=col, width=LW,
                               capstyle=tk.ROUND)
                # probability text for top-10 moves
                if p >= np.sort(policy)[-10]:
                    pc.create_text(cx, cy, text=f"{p*100:.1f}%",
                                   font=("Segoe UI", 6, "bold"),
                                   fill="#ffffff")

        # circle the chosen move in gold
        x1, y1, x2, y2, cx, cy = pol_coords(chosen)
        pc.create_oval(cx-10, cy-10, cx+10, cy+10,
                       outline="#f9a825", width=2, fill="")

        # draw dots on top
        for i in range(S+1):
            for j in range(S+1):
                x = OFF + j*SP; y = OFF + i*SP
                pc.create_oval(x-DR, y-DR, x+DR, y+DR,
                               fill="#1a1a2e", outline="")

        # update sorted list
        self._update_prob_list(rec)

    def _update_prob_list(self, rec):
        """Populate the sorted probability listbox."""
        lb = self._prob_list
        lb.delete(0, tk.END)

        policy = np.array(rec.policy, dtype=float)
        chosen_move = rec.move

        S    = self.SIZE
        half = S * (S + 1)

        # sort by probability descending, skip zero-prob entries
        order = np.argsort(policy)[::-1]
        valid_entries = [(rank, li) for rank, li in enumerate(order, 1)
                         if policy[li] > 0]

        bar_max = 20   # max bar width in chars
        p_max   = policy[valid_entries[0][1]] if valid_entries else 1.0

        for rank, li in valid_entries:
            p = policy[li]
            # H/V label and row/col
            if li < half:
                r = li // S; c = li % S
                loc = f"H r{r}c{c}"
            else:
                idx = li - half; col = idx // S; row = idx % S
                loc = f"V r{row}c{col}"

            bar_len  = max(1, round(p / p_max * bar_max))
            bar      = "█" * bar_len + "░" * (bar_max - bar_len)
            star     = "★" if li == chosen_move else " "
            
            if getattr(rec, 'visit_counts', None) is not None:
                vc = rec.visit_counts[li]
                dp = rec.child_depths[li] if getattr(rec, 'child_depths', None) else 0
                line_str = (f"{star} #{rank:2d}  line {li:2d}  {loc}  "
                            f"{bar}  {p*100:5.2f}%  ({vc:3d} visits, d={dp})")
            else:
                line_str = (f"{star} #{rank:2d}  line {li:2d}  {loc}  "
                            f"{bar}  {p*100:5.2f}%")
                
            lb.insert(tk.END, line_str)

            # colour chosen move row in the listbox
            if li == chosen_move:
                lb.itemconfig(tk.END, bg="#fff3cd", fg="#7c4a00")

        if not valid_entries:
            lb.insert(tk.END, "  (no valid moves)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best.pth.tar",
                    help="Path to AlphaZero checkpoint (default: best.pth.tar)")
    args = ap.parse_args()

    root = tk.Tk()
    app  = PolicyInspectorGUI(root, model_path=args.model)

    # keyboard shortcuts: left/right arrow keys
    root.bind("<Left>",  lambda e: app._prev_step())
    root.bind("<Right>", lambda e: app._next_step())
    root.bind("<space>", lambda e: app._on_play())

    root.mainloop()


if __name__ == "__main__":
    main()
