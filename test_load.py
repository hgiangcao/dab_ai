import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from distributed.pretrained import load_examples_from_jsonl
exs = load_examples_from_jsonl("game_logs.jsonl")
print(f"Loaded {len(exs)} examples")
