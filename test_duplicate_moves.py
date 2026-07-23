import json

def check_duplicates():
    count = 0
    with open("game_logs.jsonl", "r") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            moves = record.get("moves", [])
            if len(moves) == 60:
                if len(set(moves)) != 60:
                    print(f"Line {i} has duplicate moves! {moves}")
                    return
                count += 1
    print(f"Checked {count} full 60-move games. No duplicates found!")

if __name__ == "__main__":
    check_duplicates()
