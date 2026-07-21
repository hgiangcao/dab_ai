import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from game import DotsAndBoxesGame
from agent_interface import HumanAgent, BaseAgent
from bots.alpha_beta import AlphaBetaPlayer
from bots.mcts_heuristic import MCTSHeuristicAgent
from bots.mcts_x import MCTS100Agent, MCTS1000Agent
from bots.ucla_bot import UCLABot
import random


# A Dummy AI Agent to demonstrate how plugging in an agent works
class RandomBot(BaseAgent):
    def __init__(self, name: str = "Random Bot"):
        super().__init__(name)

    def get_move(self, game_state: DotsAndBoxesGame) -> int:
        valid_moves = game_state.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None


class DotsAndBoxesGUI:
    def __init__(self, root, size=5):
        self.root = root
        self.root.title("Dots and Boxes - Tournament Prep Portal")
        self.size = size
        
        # Initialize Game Engine
        self.game = DotsAndBoxesGame(size=self.size, starting_player=1)
        
        # Registry of available agents for the dropdowns
        self.agent_registry = {
            "Human": HumanAgent,
            "Random Bot": RandomBot,
            "Alpha-Beta Bot": AlphaBetaPlayer,
            "MCTS_100": MCTS100Agent,
            "MCTS_1000": MCTS1000Agent,
            "UCLABot": UCLABot
        }
        
        # Active Player Instances
        self.player1_agent = HumanAgent("Player 1")
        self.player2_agent = HumanAgent("Player 2")
        self.last_thinking_time = None
        self.game_moves = []
        self.game_policies = []
        self.log_saved = False

        # Visual Grid Constants
        self.DOT_RADIUS = 6
        self.SPACING = 60
        self.OFFSET = 40
        self.LINE_WIDTH = 5
        
        self._setup_widgets()
        self._draw_board_skeleton()
        self.update_ui_status()

    def _setup_widgets(self):
        # Top Panel for Controls
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Player 1 Selection (Red)
        tk.Label(control_frame, text="Player 1 (Red):", fg="red", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.p1_combo = ttk.Combobox(control_frame, values=list(self.agent_registry.keys()), state="readonly", width=12)
        self.p1_combo.set("Human")
        self.p1_combo.pack(side=tk.LEFT, padx=5)
        self.p1_combo.bind("<<ComboboxSelected>>", self._update_agents)

        # Player 2 Selection (Blue)
        tk.Label(control_frame, text="Player 2 (Blue):", fg="blue", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.p2_combo = ttk.Combobox(control_frame, values=list(self.agent_registry.keys()), state="readonly", width=12)
        self.p2_combo.set("Human")
        self.p2_combo.pack(side=tk.LEFT, padx=5)
        self.p2_combo.bind("<<ComboboxSelected>>", self._update_agents)
        
        # Reset Button
        tk.Button(control_frame, text="Reset Match", command=self.reset_game).pack(side=tk.RIGHT, padx=20)

        # Status Bar (Turn / Score trackers)
        self.status_label = tk.Label(self.root, text="", font=("Helvetica", 12, "bold"), pady=5)
        self.status_label.pack(side=tk.TOP)
        self.thinking_label = tk.Label(self.root, text="Last Bot Thinking Time: N/A", font=("Helvetica", 10, "italic"), fg="gray", pady=2)
        self.thinking_label.pack(side=tk.TOP)

        # Canvas for Game Board
        canvas_size = 2 * self.OFFSET + self.size * self.SPACING
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="#F0F0F0")
        self.canvas.pack(side=tk.BOTTOM, padx=20, pady=20)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

    def _update_agents(self, event=None):
        """Dynamically re-binds agents when selection changes in the drop box."""
        self.player1_agent = self.agent_registry[self.p1_combo.get()]("Player 1")
        self.player2_agent = self.agent_registry[self.p2_combo.get()]("Player 2")
        self.root.after(100, self.check_ai_turn)

    def _draw_board_skeleton(self):
        self.canvas.delete("all")
        # Draw Dots
        for i in range(self.size + 1):
            for j in range(self.size + 1):
                x = self.OFFSET + j * self.SPACING
                y = self.OFFSET + i * self.SPACING
                self.canvas.create_oval(x - self.DOT_RADIUS, y - self.DOT_RADIUS, 
                                        x + self.DOT_RADIUS, y + self.DOT_RADIUS, fill="black")

    def update_ui_status(self):
        # Calculate Scores from the Game Class Tracker Matrix
        p1_score = int(np.sum(self.game.b == 1))
        p2_score = int(np.sum(self.game.b == -1))
        
        p1_name = self.p1_combo.get()
        p2_name = self.p2_combo.get()
        
        if self.last_thinking_time is not None:
            self.thinking_label.config(text=f"Last Bot Thinking Time: {self.last_thinking_time:.4f}s")
        else:
            self.thinking_label.config(text="Last Bot Thinking Time: N/A")

        if self.game.is_running():
            current_color = "Red" if self.game.current_player == 1 else "Blue"
            self.status_label.config(
                text=f"Current Turn: {current_color}  |  {p1_name} {p1_score} vs {p2_score} {p2_name}",
                fg="red" if self.game.current_player == 1 else "blue"
            )
        else:
            if not self.log_saved:
                self.log_saved = True
                import os
                from game_log import save_game_log
                os.makedirs("logs", exist_ok=True)
                save_game_log("logs/gui_games.json", self.game_moves, self.game_policies, int(self.game.result))

            if self.game.result == 1:
                msg = f"🏆 Match Over! Red ({p1_name}) Wins! Final: {p1_score}-{p2_score}"
            elif self.game.result == -1:
                msg = f"🏆 Match Over! Blue ({p2_name}) Wins! Final: {p2_score}-{p1_score}"
            else:
                msg = f"🤝 Draw Match! Final Score: {p1_score}-{p2_score}"
            self.status_label.config(text=msg, fg="purple")
            messagebox.showinfo("Game Over", msg)

    def _on_canvas_click(self, event):
        """Handles human clicks on the lines."""
        if not self.game.is_running():
            return
            
        # Verify if it is currently a Human's turn
        current_agent = self.player1_agent if self.game.current_player == 1 else self.player2_agent
        if not isinstance(current_agent, HumanAgent):
            return  # Block human click inputs during AI processing turns

        # Find closest line to click coordinate
        clicked_line = self._get_closest_line_index(event.x, event.y)
        if clicked_line is not None and self.game.l[clicked_line] == 0:
            self.last_thinking_time = None
            self.process_move(clicked_line)

    def process_move(self, line_index):
        player_color = "red" if self.game.current_player == 1 else "blue"
        
        self.game_moves.append(int(line_index))
        self.game_policies.append(None)
        
        # Execute line within engine state
        self.game.execute_move(line_index)
        
        # Redraw the freshly modified board components
        self._render_drawn_lines_and_boxes()
        self.update_ui_status()
        
        # Check if the next player is an AI agent
        if self.game.is_running():
            self.root.after(100, self.check_ai_turn)

    def check_ai_turn(self):
        if not self.game.is_running():
            return
            
        current_agent = self.player1_agent if self.game.current_player == 1 else self.player2_agent
        if not isinstance(current_agent, HumanAgent):
            # Compute move using standard agent template structure
            import time
            start_time = time.perf_counter()
            ai_move = current_agent.get_move(self.game)
            self.last_thinking_time = time.perf_counter() - start_time
            
            if ai_move is not None:
                self.process_move(ai_move)

    def _render_drawn_lines_and_boxes(self):
        # Draw Box Ownership Colors
        for r in range(self.size):
            for c in range(self.size):
                if self.game.b[r][c] != 0:
                    color = "#FFCCCC" if self.game.b[r][c] == 1 else "#CCE5FF"
                    x1 = self.OFFSET + c * self.SPACING + self.DOT_RADIUS
                    y1 = self.OFFSET + r * self.SPACING + self.DOT_RADIUS
                    x2 = x1 + self.SPACING - 2 * self.DOT_RADIUS
                    y2 = y1 + self.SPACING - 2 * self.DOT_RADIUS
                    if not self.canvas.find_withtag(f"box_{r}_{c}"):
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", tags=f"box_{r}_{c}")

        # Draw Lines
        h_matrix, v_matrix = self.game.l_to_h_v(self.game.l)
        
        # Draw horizontal lines
        for r in range(self.size + 1):
            for c in range(self.size):
                if h_matrix[r][c] != 0:
                    color = "red" if h_matrix[r][c] == 1 else "blue"
                    x1 = self.OFFSET + c * self.SPACING
                    y1 = self.OFFSET + r * self.SPACING
                    x2 = x1 + self.SPACING
                    self.canvas.create_line(x1, y1, x2, y1, fill=color, width=self.LINE_WIDTH)

        # Draw vertical lines
        for c in range(self.size + 1):
            for r in range(self.size):
                if v_matrix[r][c] != 0:
                    color = "red" if v_matrix[r][c] == 1 else "blue"
                    x1 = self.OFFSET + c * self.SPACING
                    y1 = self.OFFSET + r * self.SPACING
                    y2 = y1 + self.SPACING
                    self.canvas.create_line(x1, y1, x1, y2, fill=color, width=self.LINE_WIDTH)

    def _get_closest_line_index(self, click_x, click_y):
        """Reverse maps pixel clicks into the game engine's single array line index."""
        best_dist = 15  # Click tolerance threshold
        closest_line = None
        
        # Check all lines in the engine scheme to find spatial proximity
        for line_idx in range(self.game.N_LINES):
            # Resolve bounding coordinates of line vector element
            h_matrix, v_matrix = np.zeros(self.game.N_LINES), np.zeros(self.game.N_LINES)
            h_matrix[line_idx] = 1 
            h, v = self.game.l_to_h_v(h_matrix)
            
            if np.any(h == 1): # It is a horizontal line element layout mapping
                r, c = np.where(h == 1)[0][0], np.where(h == 1)[1][0]
                lx1, ly1 = self.OFFSET + c * self.SPACING, self.OFFSET + r * self.SPACING
                lx2, ly2 = lx1 + self.SPACING, ly1
            else: # Vertical line mapping profile
                h_matrix[line_idx] = 0; v_matrix[line_idx] = 1
                _, v = self.game.l_to_h_v(v_matrix)
                r, c = np.where(v == 1)[0][0], np.where(v == 1)[1][0]
                lx1, ly1 = self.OFFSET + c * self.SPACING, self.OFFSET + r * self.SPACING
                lx2, ly2 = lx1, ly1 + self.SPACING

            # Midpoint distance validation rule
            mid_x, mid_y = (lx1 + lx2) / 2, (ly1 + ly2) / 2
            dist = np.hypot(click_x - mid_x, click_y - mid_y)
            if dist < best_dist:
                best_dist = dist
                closest_line = line_idx
                
        return closest_line

    def reset_game(self):
        self.game = DotsAndBoxesGame(size=self.size, starting_player=1)
        self.last_thinking_time = None
        self.game_moves = []
        self.game_policies = []
        self.log_saved = False
        self._draw_board_skeleton()
        self._update_agents()
        self.update_ui_status()


if __name__ == "__main__":
    root = tk.Tk()
    # Matches typical competition dimensions (5x5 boxes)
    app = DotsAndBoxesGUI(root, size=5)
    root.mainloop()