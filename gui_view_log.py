import tkinter as tk
from tkinter import ttk, messagebox
import os
import json
import math
import numpy as np
from game import DotsAndBoxesGame

class LogViewerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dots and Boxes - Game Log Viewer")
        
        # Locate available log files
        self.logs_dir = "logs"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir, exist_ok=True)
            
        self.log_files = [f for f in os.listdir(self.logs_dir) if f.endswith('.json') or f.endswith('.jsonl')]
        if not self.log_files:
            self.log_files = ["gui_games.json"]
            self.log_filepath = os.path.join(self.logs_dir, "gui_games.json")
        else:
            if "gui_games.json" in self.log_files:
                self.log_filepath = os.path.join(self.logs_dir, "gui_games.json")
            else:
                self.log_filepath = os.path.join(self.logs_dir, self.log_files[0])
        
        self.games = []
        self._load_logs()
        
        # Start with the last finished game
        self.current_game_idx = max(0, len(self.games) - 1)
        
        # Visual Grid Constants
        self.DOT_RADIUS = 6
        self.SPACING = 60
        self.OFFSET = 40
        self.LINE_WIDTH = 5

        # Initialize current game data
        self.setup_game_data()

        # Build Widgets
        self._setup_widgets()
        self.render_current_state()

    def _load_logs(self):
        if os.path.exists(self.log_filepath):
            try:
                with open(self.log_filepath, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.games.append(json.loads(line))
            except Exception as e:
                print(f"Error loading logs: {e}")

    def setup_game_data(self):
        if not self.games:
            self.moves = []
            self.winner = 0
            self.size = 5
            self.current_move_idx = 0
            self.rebuild_game_state()
            return

        game_data = self.games[self.current_game_idx]
        self.moves = game_data.get("moves", [])
        self.winner = game_data.get("winner", 0)
        
        # Infer board size
        max_move = max(self.moves) if self.moves else 0
        self.size = 5 # default fallback
        for s in range(1, 20):
            if 2 * s * (s + 1) > max_move:
                self.size = s
                break

        # Start at the final finished game state (last move)
        self.current_move_idx = len(self.moves)
        self.rebuild_game_state()

    def rebuild_game_state(self):
        # Instantiate a clean engine and play up to current_move_idx
        self.game = DotsAndBoxesGame(size=self.size, starting_player=1)
        for move in self.moves[:self.current_move_idx]:
            # Use game's execute_move to update lines, boxes and player turns
            self.game.execute_move(move)

    def _setup_widgets(self):
        # File Selection Frame
        self.file_frame = tk.Frame(self.root, pady=10, bg="#E9ECEF")
        self.file_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(self.file_frame, text="Select Log File:", font=("Helvetica", 10, "bold"), bg="#E9ECEF").pack(side=tk.LEFT, padx=15)
        self.file_combo = ttk.Combobox(self.file_frame, values=self.log_files, state="readonly", width=30)
        self.file_combo.set(os.path.basename(self.log_filepath))
        self.file_combo.pack(side=tk.LEFT, padx=5)
        self.file_combo.bind("<<ComboboxSelected>>", self.on_file_selected)
        
        self.btn_refresh = tk.Button(self.file_frame, text="🔄 Refresh", command=self.refresh_log_files, font=("Helvetica", 9))
        self.btn_refresh.pack(side=tk.RIGHT, padx=15)

        # Top Frame for game selection
        self.top_frame = tk.Frame(self.root, pady=10)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_prev_game = tk.Button(self.top_frame, text="◀ Prev Game", command=self.prev_game, font=("Helvetica", 10, "bold"))
        self.btn_prev_game.pack(side=tk.LEFT, padx=20)

        self.game_status_label = tk.Label(self.top_frame, text="", font=("Helvetica", 11, "bold"))
        self.game_status_label.pack(side=tk.LEFT, expand=True)

        self.btn_next_game = tk.Button(self.top_frame, text="Next Game ▶", command=self.next_game, font=("Helvetica", 10, "bold"))
        self.btn_next_game.pack(side=tk.RIGHT, padx=20)

        # Middle Canvas
        canvas_size = 2 * self.OFFSET + self.size * self.SPACING
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="#F8F9FA", highlightthickness=1, highlightbackground="#DEE2E6")
        self.canvas.pack(side=tk.TOP, padx=20, pady=10)

        # Bottom Frame for move navigation
        self.bottom_frame = tk.Frame(self.root, pady=10)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_prev_move = tk.Button(self.bottom_frame, text="◀ Prev Move", command=self.prev_move, font=("Helvetica", 10))
        self.btn_prev_move.pack(side=tk.LEFT, padx=20)

        self.move_status_label = tk.Label(self.bottom_frame, text="", font=("Helvetica", 10, "italic"))
        self.move_status_label.pack(side=tk.LEFT, expand=True)

        self.btn_next_move = tk.Button(self.bottom_frame, text="Next Move ▶", command=self.next_move, font=("Helvetica", 10))
        self.btn_next_move.pack(side=tk.RIGHT, padx=20)

    def render_current_state(self):
        # Update Canvas size dynamic change if game sizes differ
        canvas_size = 2 * self.OFFSET + self.size * self.SPACING
        self.canvas.config(width=canvas_size, height=canvas_size)

        # Clear and redraw skeleton
        self.canvas.delete("all")
        for i in range(self.size + 1):
            for j in range(self.size + 1):
                x = self.OFFSET + j * self.SPACING
                y = self.OFFSET + i * self.SPACING
                self.canvas.create_oval(x - self.DOT_RADIUS, y - self.DOT_RADIUS, 
                                        x + self.DOT_RADIUS, y + self.DOT_RADIUS, fill="#212529")

        # Draw box backgrounds
        for r in range(self.size):
            for c in range(self.size):
                if self.game.b[r][c] != 0:
                    color = "#FFCCCC" if self.game.b[r][c] == 1 else "#CCE5FF"
                    x1 = self.OFFSET + c * self.SPACING + self.DOT_RADIUS
                    y1 = self.OFFSET + r * self.SPACING + self.DOT_RADIUS
                    x2 = x1 + self.SPACING - 2 * self.DOT_RADIUS
                    y2 = y1 + self.SPACING - 2 * self.DOT_RADIUS
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

        # Draw lines
        h_matrix, v_matrix = self.game.l_to_h_v(self.game.l)
        # Horizontal lines
        for r in range(self.size + 1):
            for c in range(self.size):
                if h_matrix[r][c] != 0:
                    color = "red" if h_matrix[r][c] == 1 else "blue"
                    x1 = self.OFFSET + c * self.SPACING
                    y1 = self.OFFSET + r * self.SPACING
                    x2 = x1 + self.SPACING
                    self.canvas.create_line(x1, y1, x2, y1, fill=color, width=self.LINE_WIDTH)

        # Vertical lines
        for c in range(self.size + 1):
            for r in range(self.size):
                if v_matrix[r][c] != 0:
                    color = "red" if v_matrix[r][c] == 1 else "blue"
                    x1 = self.OFFSET + c * self.SPACING
                    y1 = self.OFFSET + r * self.SPACING
                    y2 = y1 + self.SPACING
                    self.canvas.create_line(x1, y1, x1, y2, fill=color, width=self.LINE_WIDTH)

        # Highlight the last drawn line if applicable
        if self.current_move_idx > 0:
            last_move = self.moves[self.current_move_idx - 1]
            h_dummy, v_dummy = np.zeros(self.game.N_LINES), np.zeros(self.game.N_LINES)
            h_dummy[last_move] = 1
            h, v = self.game.l_to_h_v(h_dummy)
            if np.any(h == 1):
                r, c = np.where(h == 1)[0][0], np.where(h == 1)[1][0]
                x1, y1 = self.OFFSET + c * self.SPACING, self.OFFSET + r * self.SPACING
                x2, y2 = x1 + self.SPACING, y1
            else:
                h_dummy[last_move] = 0; v_dummy[last_move] = 1
                _, v = self.game.l_to_h_v(v_dummy)
                r, c = np.where(v == 1)[0][0], np.where(v == 1)[1][0]
                x1, y1 = self.OFFSET + c * self.SPACING, self.OFFSET + r * self.SPACING
                x2, y2 = x1, y1 + self.SPACING
            # Draw a subtle highlight border/glow around it
            self.canvas.create_line(x1, y1, x2, y2, fill="#FFC107", width=2)

        # Update Labels
        p1_score = int(np.sum(self.game.b == 1))
        p2_score = int(np.sum(self.game.b == -1))
        
        winner_str = "Draw" if self.winner == 0 else ("Red" if self.winner == 1 else "Blue")
        self.game_status_label.config(
            text=f"Game {self.current_game_idx + 1} / {len(self.games)}  |  Winner: {winner_str}"
        )
        
        self.move_status_label.config(
            text=f"Move {self.current_move_idx} / {len(self.moves)}  |  Score -> Red: {p1_score} vs Blue: {p2_score}"
        )

        # Button states
        self.btn_prev_game.config(state="normal" if self.current_game_idx > 0 else "disabled")
        self.btn_next_game.config(state="normal" if self.current_game_idx < len(self.games) - 1 else "disabled")
        self.btn_prev_move.config(state="normal" if self.current_move_idx > 0 else "disabled")
        self.btn_next_move.config(state="normal" if self.current_move_idx < len(self.moves) else "disabled")

    def prev_game(self):
        if self.current_game_idx > 0:
            self.current_game_idx -= 1
            self.setup_game_data()
            self.render_current_state()

    def next_game(self):
        if self.current_game_idx < len(self.games) - 1:
            self.current_game_idx += 1
            self.setup_game_data()
            self.render_current_state()

    def next_move(self):
        if self.current_move_idx < len(self.moves):
            self.current_move_idx += 1
            self.rebuild_game_state()
            self.render_current_state()

    def prev_move(self):
        if self.current_move_idx > 0:
            self.current_move_idx -= 1
            self.rebuild_game_state()
            self.render_current_state()

    def on_file_selected(self, event=None):
        selected_file = self.file_combo.get()
        self.log_filepath = os.path.join(self.logs_dir, selected_file)
        self.games = []
        self._load_logs()
        
        # Start with the last finished game if any, otherwise 0
        self.current_game_idx = max(0, len(self.games) - 1)
        self.setup_game_data()
        self.render_current_state()

    def refresh_log_files(self):
        self.log_files = [f for f in os.listdir(self.logs_dir) if f.endswith('.json') or f.endswith('.jsonl')]
        if not self.log_files:
            self.log_files = ["gui_games.json"]
        self.file_combo.config(values=self.log_files)
        current_basename = os.path.basename(self.log_filepath)
        if current_basename in self.log_files:
            self.file_combo.set(current_basename)
        else:
            self.file_combo.set(self.log_files[0])
            self.on_file_selected()

if __name__ == "__main__":
    root = tk.Tk()
    app = LogViewerGUI(root)
    root.mainloop()
