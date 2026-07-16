import os
import glob

def merge_logs():
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(project_root, "game_logs")
    output_file = os.path.join(project_root, "game_logs.jsonl")
    
    # Find all .jsonl files in the game_logs directory
    log_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    
    if not log_files:
        print(f"No .jsonl files found in {input_dir}")
        return
    
    print(f"Found {len(log_files)} files to merge:")
    for lf in log_files:
        print(f"  - {os.path.basename(lf)} ({os.path.getsize(lf) / (1024*1024):.2f} MB)")
        
    print(f"\nMerging into {output_file}...")
    
    total_lines = 0
    with open(output_file, "w") as outfile:
        for lf in log_files:
            print(f"Processing {os.path.basename(lf)}...")
            with open(lf, "r") as infile:
                for line in infile:
                    if line.strip():
                        outfile.write(line)
                        total_lines += 1
                        
    print(f"\nSuccessfully merged {total_lines:,} games into {output_file} ({os.path.getsize(output_file) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    merge_logs()
