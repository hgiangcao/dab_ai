import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from game import DotsAndBoxesGame
from agent_interface import HumanAgent, BaseAgent
from bots.alpha_beta import AlphaBetaPlayer
from bots.mcts_heuristic import MCTSHeuristicAgent
from bots.mcts_x import MCTS100Agent, MCTS1000Agent, MCTSGAgent
from bots.greedy import GreedyPlayer
from bots.greedy_improve import GreedyChainPlayer
from bots.ucla_bot import UCLABot

class RandomBot(BaseAgent):
    def __init__(self, name: str = "Random Bot"):
        super().__init__(name)

    def get_move(self, game_state: DotsAndBoxesGame) -> int:
        valid_moves = game_state.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None

class AlphaZeroAgent(BaseAgent):
    def __init__(self, name: str = "AlphaZero"):
        super().__init__(name)
        import os
        import torch
        import config
        from model import NNetWrapper, dotdict
        from mcts import MCTS
        import model_manager
        
        self.eval_args = dotdict({
            'lr': config.LEARNING_RATE,
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'num_channels': 256,
            'num_res_blocks': 10, 
            'l2_reg': 1e-4,
            'n_simulations': config.MCTS_NUM_SIMULATIONS,
            'c_puct': config.MCTS_C_PUCT,
            'dirichlet_eps': 0.0,
            'dirichlet_alpha': config.MCTS_DIRICHLET_ALPHA,
            'device': 'cpu'
        })
        
        self.game_ref = DotsAndBoxesGame(size=5)
        self.net = NNetWrapper(self.game_ref, self.eval_args)
        
        best_path = model_manager.get_best_model_path()
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location='cpu', weights_only=False)
            self.net.nnet.load_state_dict(state['state_dict'] if 'state_dict' in state else state)
            print(f"Loaded AlphaZero model from {best_path}")
        else:
            print(f"Warning: AlphaZero model not found at {best_path}")
            
        self.net.nnet.eval()
        self.mcts = MCTS(self.net, self.eval_args)

    def get_move(self, game_state: DotsAndBoxesGame) -> int:
        import time
        t0 = time.time()
        pi = self.mcts.play(game_state, temp=0)
        t1 = time.time()
        
        self.last_simuls = self.eval_args.n_simulations
        self.last_depth = getattr(self.mcts, 'max_depth_reached', 0)
        self.last_time = t1 - t0
        return int(np.argmax(pi))

class WebIntegratedGUI:
    def __init__(self, root, size=5):
        self.root = root
        self.root.title("Dots and Boxes - Web Integration")
        self.size = size
        
        # Initialize Game Engine
        self.game = DotsAndBoxesGame(size=self.size, starting_player=1)
        self.drawn_on_web = set()
        self.game_moves = []
        
        self.agent_registry = {
            "Human": lambda _: HumanAgent("You"),
            "Random": lambda _: RandomBot("Random"),
            "Greedy": lambda _: GreedyPlayer("Greedy"),
            "GreedyChain": lambda _: GreedyChainPlayer("GreedyChain"),
            "AlphaBeta_0.1s": lambda _: AlphaBetaPlayer(name="AlphaBeta_0.1s", time_limit=0.1),
            "AlphaBeta_1s": lambda _: AlphaBetaPlayer(name="AlphaBeta_1s", time_limit=10.0),
            "MCTS_0.1s": lambda _: MCTSGAgent(name="MCTS_0.1s", time_limit=0.1),
             "MCTS_1s": lambda _: MCTSGAgent(name="MCTS_1s", n_simulations=1000),
            "AlphaZero": lambda _: AlphaZeroAgent("AlphaZero"),
            "UCLABot": lambda _: UCLABot("UCLA JS Bot")
        }
        self.local_agent = HumanAgent("You")
        self.autoplay_active = False
        
        # Visual Grid Constants
        self.DOT_RADIUS = 6
        self.SPACING = 60
        self.OFFSET = 40
        self.LINE_WIDTH = 5

        # Initialize Web Driver
        self._init_web_driver()

        # Setup GUI Widgets
        self._setup_widgets()
        self._draw_board_skeleton()
        self.update_ui_status()
        
        # Start synchronization loop
        self.sync_active = True
        self.sync_state()

    def _init_web_driver(self):
        self.driver = webdriver.Chrome()
        self.driver.get("https://www.math.ucla.edu/~tom/Games/dots&boxes.html")
        
        wait = WebDriverWait(self.driver, 10)
        
        alert = wait.until(EC.alert_is_present())
        alert.send_keys(str(self.size))
        alert.accept()
        
        alert = wait.until(EC.alert_is_present())
        alert.send_keys(str(self.size))
        alert.accept()
        
    def _setup_widgets(self):
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        bot_frame = tk.Frame(control_frame)
        bot_frame.pack(side=tk.TOP, pady=5)
        
        tk.Label(bot_frame, text="Your Agent (Blue):", fg="blue", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.agent_combo = ttk.Combobox(bot_frame, values=list(self.agent_registry.keys()), state="readonly", width=15)
        self.agent_combo.set("Human")
        self.agent_combo.pack(side=tk.LEFT, padx=5)
        self.agent_combo.bind("<<ComboboxSelected>>", self._update_agent)
        
        self.start_btn = tk.Button(bot_frame, text="Start Auto-play", command=self.start_autoplay)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.reset_btn = tk.Button(bot_frame, text="New Match", command=self.reset_match)
        self.reset_btn.pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(control_frame, text="Initializing...", font=("Helvetica", 12, "bold"), pady=5)
        self.status_label.pack(side=tk.TOP)
        
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=20, pady=20)
        
        canvas_size = 2 * self.OFFSET + self.size * self.SPACING
        self.canvas = tk.Canvas(left_frame, width=canvas_size, height=canvas_size, bg="#F0F0F0")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.Y)
        
        tk.Label(right_frame, text="Move Log", font=("Helvetica", 10, "bold")).pack(side=tk.TOP)
        
        scrollbar = tk.Scrollbar(right_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_listbox = tk.Listbox(right_frame, yscrollcommand=scrollbar.set, width=30, height=20, font=("Consolas", 10))
        self.log_listbox.pack(side=tk.LEFT, fill=tk.BOTH)
        scrollbar.config(command=self.log_listbox.yview)

    def _update_agent(self, event=None):
        agent_name = self.agent_combo.get()
        self.local_agent = self.agent_registry[agent_name](agent_name)
        if isinstance(self.local_agent, HumanAgent):
            self.autoplay_active = False

    def start_autoplay(self):
        if not isinstance(self.local_agent, HumanAgent):
            self.autoplay_active = True
            self.check_local_ai_turn()

    def check_local_ai_turn(self):
        if not self.game.is_running() or not self.sync_active or not self.autoplay_active:
            return
            
        if self.game.current_player == 1 and not isinstance(self.local_agent, HumanAgent):
            move_idx = self.local_agent.get_move(self.game)
            if hasattr(self.local_agent, 'last_simuls') and move_idx is not None:
                stats_msg = f"  > MCTS: depth {self.local_agent.last_depth}, {self.local_agent.last_simuls} sims, {self.local_agent.last_time:.2f}s"
                self.log_listbox.insert(tk.END, stats_msg)
                self.log_listbox.see(tk.END)
            if move_idx is not None:
                self._process_local_move(move_idx)

    def reset_match(self):
        # 1. Dismiss any end-game alerts on the web
        try:
            alert = self.driver.switch_to.alert
            alert.accept()
        except:
            pass
            
        # 2. Reload the web game
        self.driver.get("https://www.math.ucla.edu/~tom/Games/dots&boxes.html")
        wait = WebDriverWait(self.driver, 10)
        
        try:
            alert = wait.until(EC.alert_is_present())
            alert.send_keys(str(self.size))
            alert.accept()
            
            alert = wait.until(EC.alert_is_present())
            alert.send_keys(str(self.size))
            alert.accept()
        except:
            pass
            
        # 3. Reset local state
        self.game = DotsAndBoxesGame(size=self.size, starting_player=1)
        self.drawn_on_web = set()
        self.game_moves = []
        self.autoplay_active = False
        self.log_listbox.delete(0, tk.END)
        
        self._draw_board_skeleton()
        self.update_ui_status()

    def _draw_board_skeleton(self):
        self.canvas.delete("all")
        for i in range(self.size + 1):
            for j in range(self.size + 1):
                x = self.OFFSET + j * self.SPACING
                y = self.OFFSET + i * self.SPACING
                self.canvas.create_oval(x - self.DOT_RADIUS, y - self.DOT_RADIUS, 
                                        x + self.DOT_RADIUS, y + self.DOT_RADIUS, fill="black")

    def _process_local_move(self, line_idx):
        web_name = self._line_index_to_web_name(line_idx)
        
        self.game.execute_move(line_idx)
        self.game_moves.append(line_idx)
        self.drawn_on_web.add(web_name)
        
        agent_name = "You" if isinstance(self.local_agent, HumanAgent) else self.local_agent.name
        self.log_listbox.insert(tk.END, f"Blue ({agent_name}): {web_name}")
        self.log_listbox.see(tk.END)
        self._render_drawn_lines_and_boxes()
        self.update_ui_status()
        
        try:
            element = self.driver.find_element(By.NAME, web_name)
            element.click()
        except Exception as e:
            print(f"Error clicking web element {web_name}: {e}")
            
        if self.game.current_player == 1 and self.autoplay_active:
            self.root.after(1000, self.check_local_ai_turn)

    def _on_canvas_click(self, event):
        if not self.game.is_running():
            return
            
        if self.game.current_player != 1:
            return
            
        if not isinstance(self.local_agent, HumanAgent):
            return
            
        clicked_line = self._get_closest_line_index(event.x, event.y)
        if clicked_line is not None and self.game.l[clicked_line] == 0:
            self._process_local_move(clicked_line)

    def sync_state(self):
        if not self.sync_active:
            return
            
        try:
            images = self.driver.find_elements(By.TAG_NAME, "img")
            current_web_lines = []
            for img in images:
                name = img.get_attribute("name")
                src = img.get_attribute("src")
                if name and (name.startswith("he") or name.startswith("ve")):
                    if "blank.gif" not in src:
                        current_web_lines.append(name)
            
            new_lines = []
            for name in current_web_lines:
                if name not in self.drawn_on_web:
                    new_lines.append(name)
            
            if new_lines:
                temp_l = np.copy(self.game.l)
                remaining_lines = set(new_lines)
                ordered_lines = []
                
                while remaining_lines:
                    progress = False
                    for name in list(remaining_lines):
                        line_idx = self._web_name_to_line_index(name)
                        captures = False
                        for box in self.game.get_boxes_of_line(line_idx):
                            box_lines = self.game.get_lines_of_box(box)
                            if np.count_nonzero(temp_l[box_lines]) == 3:
                                captures = True
                                break
                                
                        if captures:
                            ordered_lines.append(name)
                            remaining_lines.remove(name)
                            temp_l[line_idx] = -1
                            progress = True
                            
                    if not progress:
                        ordered_lines.extend(list(remaining_lines))
                        break
                        
                for name in ordered_lines:
                    self.drawn_on_web.add(name)
                    line_idx = self._web_name_to_line_index(name)
                    if self.game.l[line_idx] == 0:
                        self.game.current_player = -1
                        self.game.execute_move(line_idx)
                        self.game_moves.append(line_idx)
                        self.log_listbox.insert(tk.END, f"Red (Web Bot): {name}")
                        self.log_listbox.see(tk.END)
                
                self._render_drawn_lines_and_boxes()
                self.update_ui_status()
                
                if self.game.current_player == 1 and self.autoplay_active:
                    self.root.after(500, self.check_local_ai_turn)
                
        except Exception as e:
            pass
            
        if self.sync_active:
            self.root.after(500, self.sync_state)

    def update_ui_status(self):
        p1_score = int(np.sum(self.game.b == 1))
        p2_score = int(np.sum(self.game.b == -1))
        
        if self.game.is_running():
            current_color = "Blue" if self.game.current_player == 1 else "Red"
            self.status_label.config(
                text=f"Turn: {current_color}  |  Blue: {p1_score} vs Red: {p2_score}",
                fg="blue" if self.game.current_player == 1 else "red"
            )
        else:
            if self.game.result == 1:
                msg = f"Match Over! Blue Wins! Final: {p1_score}-{p2_score}"
            elif self.game.result == -1:
                msg = f"Match Over! Red Wins! Final: {p2_score}-{p1_score}"
            else:
                msg = f"Draw Match! Final Score: {p1_score}-{p2_score}"
            self.status_label.config(text=msg, fg="purple")

    def _render_drawn_lines_and_boxes(self):
        for r in range(self.size):
            for c in range(self.size):
                if self.game.b[r][c] != 0:
                    color = "#CCE5FF" if self.game.b[r][c] == 1 else "#FFCCCC"
                    x1 = self.OFFSET + c * self.SPACING + self.DOT_RADIUS
                    y1 = self.OFFSET + r * self.SPACING + self.DOT_RADIUS
                    x2 = x1 + self.SPACING - 2 * self.DOT_RADIUS
                    y2 = y1 + self.SPACING - 2 * self.DOT_RADIUS
                    if not self.canvas.find_withtag(f"box_{r}_{c}"):
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", tags=f"box_{r}_{c}")

        h_matrix, v_matrix = self.game.l_to_h_v(self.game.l)
        
        for r in range(self.size + 1):
            for c in range(self.size):
                if h_matrix[r][c] != 0:
                    color = "blue" if h_matrix[r][c] == 1 else "red"
                    x1 = self.OFFSET + c * self.SPACING
                    y1 = self.OFFSET + r * self.SPACING
                    x2 = x1 + self.SPACING
                    self.canvas.create_line(x1, y1, x2, y1, fill=color, width=self.LINE_WIDTH)

        for c in range(self.size + 1):
            for r in range(self.size):
                if v_matrix[r][c] != 0:
                    color = "blue" if v_matrix[r][c] == 1 else "red"
                    x1 = self.OFFSET + c * self.SPACING
                    y1 = self.OFFSET + r * self.SPACING
                    y2 = y1 + self.SPACING
                    self.canvas.create_line(x1, y1, x1, y2, fill=color, width=self.LINE_WIDTH)

    def _get_closest_line_index(self, click_x, click_y):
        best_dist = 15
        closest_line = None
        
        for line_idx in range(self.game.N_LINES):
            h_matrix, v_matrix = np.zeros(self.game.N_LINES), np.zeros(self.game.N_LINES)
            h_matrix[line_idx] = 1 
            h, v = self.game.l_to_h_v(h_matrix)
            
            if np.any(h == 1):
                r, c = np.where(h == 1)[0][0], np.where(h == 1)[1][0]
                lx1, ly1 = self.OFFSET + c * self.SPACING, self.OFFSET + r * self.SPACING
                lx2, ly2 = lx1 + self.SPACING, ly1
            else:
                h_matrix[line_idx] = 0; v_matrix[line_idx] = 1
                _, v = self.game.l_to_h_v(v_matrix)
                r, c = np.where(v == 1)[0][0], np.where(v == 1)[1][0]
                lx1, ly1 = self.OFFSET + c * self.SPACING, self.OFFSET + r * self.SPACING
                lx2, ly2 = lx1, ly1 + self.SPACING

            mid_x, mid_y = (lx1 + lx2) / 2, (ly1 + ly2) / 2
            dist = np.hypot(click_x - mid_x, click_y - mid_y)
            if dist < best_dist:
                best_dist = dist
                closest_line = line_idx
                
        return closest_line

    def _line_index_to_web_name(self, line_index):
        if line_index < (self.game.N_LINES / 2):
            r = line_index // self.size
            c = line_index % self.size
            return f"he{r}{c}"
        else:
            rem = line_index - int(self.game.N_LINES / 2)
            c = rem // self.size
            r = rem % self.size
            return f"ve{r}{c}"
            
    def _web_name_to_line_index(self, web_name):
        r = int(web_name[2])
        c = int(web_name[3])
        if web_name.startswith("he"):
            return r * self.size + c
        elif web_name.startswith("ve"):
            return int(self.game.N_LINES / 2) + c * self.size + r

    def on_close(self):
        self.sync_active = False
        try:
            self.driver.quit()
        except:
            pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebIntegratedGUI(root, size=5)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
