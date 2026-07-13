import json

def save_game_log(filepath: str, moves: list, policies: list, winner: int):
    """
    Appends a single game to the log.
    moves: list of integers (the line indices played)
    policies: list of lists (the MCTS probability vectors for each move)
    winner: 1, -1, or 0
    """
    record = {
        "winner": winner,
        "moves": moves,
        "policies": policies 
    }
    
    with open(filepath, 'a') as f:
        f.write(json.dumps(record) + '\n')

def load_logs_generator(filepath: str):
    """
    Yields one game at a time to prevent RAM overflow during training.
    """
    with open(filepath, 'r') as f:
        for line in f:
            yield json.loads(line)
