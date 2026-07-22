import time
import os
import threading
import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import UnexpectedAlertPresentException
import re

from game import DotsAndBoxesGame
from bots.alpha_beta import AlphaBetaPlayer
from bots.mcts_x import MCTSGAgent
from bots.greedy import GreedyPlayer
from bots.greedy_improve import GreedyChainPlayer
from bots.ucla_bot import UCLABot, UCLABot_v2, UCLABot_v3
from web_gui_game import RandomBot, AlphaZeroAgent

class WebTournament:
    def __init__(self, size=5, n_games=20):
        self.size = size
        self.n_games = n_games
        
        self.agent_registry = {
            # "Random": lambda _: RandomBot("Random"),
            # "Greedy": lambda _: GreedyPlayer("Greedy"),
            # "GreedyChain": lambda _: GreedyChainPlayer("GreedyChain"),
            # "AlphaBeta_1s": lambda _: AlphaBetaPlayer(name="AlphaBeta_1s", time_limit=1),
            # "AlphaBeta_10s": lambda _: AlphaBetaPlayer(name="AlphaBeta_10s", time_limit=10.0),
            # "UCLABot": lambda _: UCLABot("UCLA JS Bot"),
            # "UCLABot_v2": lambda _: UCLABot_v2("UCLA JS Bot v2"),
            #"UCLABot_v3": lambda _: UCLABot_v3("UCLA JS Bot v3"),
            # "UCLAGreedyBot": lambda _: __import__('bots.ucla_bot_heuristic', fromlist=['UCLAGreedyBot']).UCLAGreedyBot("UCLA Greedy"),
            "UCLAAlphaBeta": lambda _: __import__('bots.ucla_alpha_beta', fromlist=['UCLAAlphaBeta']).UCLAAlphaBeta("UCLA AlphaBeta"),
            # "MCTS_1s": lambda _: MCTSGAgent(name="MCTS_1s", time_limit=1),
            # "AlphaZero": lambda _: AlphaZeroAgent("AlphaZero", n_simulations=500),
            # "AlphaZero_1000": lambda _: AlphaZeroAgent("AlphaZero_1000", n_simulations=1000),
        }
        
        # Thread safety utilities
        self.thread_local = threading.local()
        self.drivers = []
        self.driver_lock = threading.Lock()
        self.print_lock = threading.Lock()
        
    def get_driver(self):
        if not hasattr(self.thread_local, "driver"):
            options = webdriver.ChromeOptions()
            # options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--log-level=3')
            driver = webdriver.Chrome(options=options)
            self.thread_local.driver = driver
            with self.driver_lock:
                self.drivers.append(driver)
        return self.thread_local.driver

    def _line_index_to_web_name(self, game, line_index):
        if line_index < (game.N_LINES / 2):
            r = line_index // self.size
            c = line_index % self.size
            return f"he{r}{c}"
        else:
            rem = line_index - int(game.N_LINES / 2)
            c = rem // self.size
            r = rem % self.size
            return f"ve{r}{c}"
            
    def _web_name_to_line_index(self, game, web_name):
        r = int(web_name[2])
        c = int(web_name[3])
        if web_name.startswith("he"):
            return r * self.size + c
        elif web_name.startswith("ve"):
            return int(game.N_LINES / 2) + c * self.size + r

    def reset_web_game(self, driver):
        try:
            alert = driver.switch_to.alert
            alert.accept()
        except:
            pass
            
        local_path = os.path.abspath("Dots and Boxes.html").replace("\\", "/")
        local_url = f"file:///{local_path}"
        
        for _ in range(3):
            try:
                driver.get(local_url)
                break
            except Exception:
                time.sleep(0.5)

    def play_game(self, bot_name, game_idx):
        try:
            driver = self.get_driver()
            self.reset_web_game(driver)
            time.sleep(0.5) # Wait for DOM to fully load
            
            game = DotsAndBoxesGame(size=self.size, starting_player=1)
            
            try:
                local_agent = self.agent_registry[bot_name](bot_name)
            except Exception as e:
                return bot_name, game_idx, "Error", 0
                
            drawn_on_web = set()
            
            while game.is_running():
                if game.current_player == 1:
                    move_idx = local_agent.get_move(game)
                    web_name = self._line_index_to_web_name(game, move_idx)
                    
                    game.execute_move(move_idx)
                    drawn_on_web.add(web_name)
                    
                    try:
                        element = driver.find_element(By.NAME, web_name)
                        element.click()
                    except Exception as e:
                        pass
                else:
                    polled = False
                    timeout = time.time() + 30 # 30s timeout for web bot to make a move
                    while not polled:
                        if time.time() > timeout:
                            return bot_name, game_idx, "Error", 0
                            
                        try:
                            current_web_lines = driver.execute_script("""
                                return Array.from(document.getElementsByTagName('img'))
                                    .filter(img => img.name && (img.name.startsWith('he') || img.name.startsWith('ve')) && !img.src.includes('blank.gif'))
                                    .map(img => img.name);
                            """)
                                        
                            new_lines = [name for name in current_web_lines if name not in drawn_on_web]
                            
                            if new_lines:
                                temp_l = np.copy(game.l)
                                remaining_lines = set(new_lines)
                                ordered_lines = []
                                
                                while remaining_lines:
                                    progress = False
                                    for name in list(remaining_lines):
                                        line_idx = self._web_name_to_line_index(game, name)
                                        captures = False
                                        for box in game.get_boxes_of_line(line_idx):
                                            box_lines = game.get_lines_of_box(box)
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
                                    drawn_on_web.add(name)
                                    line_idx = self._web_name_to_line_index(game, name)
                                    if game.l[line_idx] == 0:
                                        game.current_player = -1
                                        game.execute_move(line_idx)
                                polled = True
                            else:
                                time.sleep(0.05)
                        except Exception as e:
                            time.sleep(0.05)
                            continue
                            
            p1_score = int(np.sum(game.b == 1))
            p2_score = int(np.sum(game.b == -1))
            margin = p1_score - p2_score
            
            result = "Win" if p1_score > p2_score else ("Loss" if p2_score > p1_score else "Draw")
            return bot_name, game_idx, result, margin
            
        except Exception as outer_e:
            import traceback
            traceback.print_exc()
            return bot_name, game_idx, "Error", 0

    def run_tournament(self, specific_bot=None):
        if specific_bot:
            if specific_bot in self.agent_registry:
                self.agent_registry = {specific_bot: self.agent_registry[specific_bot]}
            else:
                print(f"Bot '{specific_bot}' not found in registry.")
                return
                
        results = {bot: {"Wins": 0, "Losses": 0, "Draws": 0, "TotalMargin": 0} for bot in self.agent_registry}
        
        log_file = open("tournament_web_bot_log.txt", "w")
        log_file.write(f"Tournament: Local Bots vs Web Bot (N={self.n_games} games each, Sequential)\n")
        log_file.write("="*60 + "\n")
        
        print("Starting Tournament, running bots sequentially...")
        
        # Prepare all match tasks
        tasks = []
        for bot_name in self.agent_registry.keys():
            for i in range(self.n_games):
                tasks.append((bot_name, i))
                
        # Initialize tqdm for each bot
        pbars = {}
        for i, bot_name in enumerate(self.agent_registry.keys()):
            pbars[bot_name] = tqdm(total=self.n_games, desc=f"{bot_name[:15]:<15}", position=i, leave=True)
            
        bot_games_played = {bot: 0 for bot in self.agent_registry}
                
        # Execute tasks sequentially
        for bot_name, game_idx in tasks:
            try:
                ret_bot_name, ret_game_idx, res, margin = self.play_game(bot_name, game_idx)
                if res == "Win":
                    results[bot_name]["Wins"] += 1
                elif res == "Loss":
                    results[bot_name]["Losses"] += 1
                elif res == "Draw":
                    results[bot_name]["Draws"] += 1
                    
                if res != "Error" and res is not None:
                    results[bot_name]["TotalMargin"] += margin
                    
                bot_games_played[bot_name] += 1
                played = bot_games_played[bot_name]
                wr = results[bot_name]["Wins"] / played if played > 0 else 0
                
                pbars[bot_name].set_postfix({"WR": f"{wr:.1%}"})
                pbars[bot_name].update(1)
            except Exception as e:
                tqdm.write(f"Exception in game {bot_name} {game_idx}: {e}")
                    
        print("\n" * len(self.agent_registry)) # Clear lines so following prints don't overwrite pbars
                    
        # Cleanup drivers
        with self.driver_lock:
            for d in self.drivers:
                try:
                    d.quit()
                except:
                    pass
            self.drivers.clear()
            
        # Compile final stats
        final_stats = {}
        for bot_name, stats in results.items():
            wins = stats["Wins"]
            losses = stats["Losses"]
            draws = stats["Draws"]
            total_margin = stats["TotalMargin"]
            
            if self.n_games > 0:
                win_rate = wins / self.n_games
                draw_rate = draws / self.n_games
                avg_margin = total_margin / self.n_games
            else:
                win_rate = draw_rate = avg_margin = 0.0
                
            final_stats[bot_name] = {
                "Win Rate": win_rate,
                "Draw Rate": draw_rate,
                "Avg Margin": avg_margin,
                "Wins": wins,
                "Losses": losses,
                "Draws": draws
            }
            
            log_line = f"{bot_name:<15} | Wins: {wins:<2} | Losses: {losses:<2} | Draws: {draws:<2} | Win Rate: {win_rate:.2f} | Avg Margin: {avg_margin:.2f}\n"
            log_file.write(log_line)
            print(log_line.strip())
            
        log_file.close()
        self.generate_heatmap(final_stats)
        
    def generate_heatmap(self, results):
        df = pd.DataFrame.from_dict(results, orient='index')
        df_heatmap = df[['Win Rate', 'Draw Rate', 'Avg Margin']]
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_heatmap, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=.5)
        plt.title("Tournament Results vs UCLA Web Bot")
        plt.tight_layout()
        plt.savefig("tournament_vs_web_bot_heatmap.png", dpi=150)
        print("Heatmap saved to tournament_vs_web_bot_heatmap.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run parallel tournament against web bot")
    parser.add_argument("--games", type=int, default=20, help="Number of games per bot")
    parser.add_argument("--bot", type=str, default=None, help="Specific bot to test")
    args = parser.parse_args()
    
    tourney = WebTournament(size=5, n_games=args.games)
    tourney.run_tournament(specific_bot=args.bot)
