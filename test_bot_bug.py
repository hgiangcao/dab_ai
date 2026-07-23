import traceback
from DotsAndBoxesAlphaZero import DotsAndBoxesGame
from bots.simple_bot_v2 import SimpleBotV2
import copy

def run_test():
    try:
        baseline = SimpleBotV2(name="SimpleBotV2")
        game = DotsAndBoxesGame(size=5, starting_player=1, early_stopping=True)

        for _ in range(50): # try 50 games
            game = DotsAndBoxesGame(size=5, starting_player=1, early_stopping=True)
            while game.is_running():
                move = baseline.get_move(copy.deepcopy(game))
                game.execute_move(move)
        print("Success, no error.")
    except Exception as e:
        print("Error!")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
