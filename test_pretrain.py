import sys
import os
import torch
PROJECT_ROOT = "/home/user/dab_ai"
sys.path.append(PROJECT_ROOT)
from distributed.pretrained import load_examples_from_jsonl, GameDataset
dataset = GameDataset(load_examples_from_jsonl(os.path.join(PROJECT_ROOT, "game_logs_bot.jsonl"), 5))
print("Dataset size:", len(dataset))
if len(dataset) > 0:
    print("Example 0:", dataset[0])
    b, p, v = dataset[0]
    print("board mean:", b.mean())
    print("policy max:", p.max())
    print("value:", v)
