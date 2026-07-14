import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil
import time

from client import ServerClient
from selfplay import SelfPlayGenerator


import argparse

def main():
    parser = argparse.ArgumentParser(description="Distributed Self-Play Worker")
    parser.add_argument("--server", type=str, default="172.16.2.31:8000", help="Server address (IP:port)")
    parser.add_argument("--worker", type=str, default="worker01", help="Worker name identifier")
    parser.add_argument("--games", type=int, default=100, help="Number of games per batch")
    args = parser.parse_args()

    # Create worker-specific local directories to prevent collision when running multiple workers locally
    local_dir = f"./worker_data_{args.worker}"
    model_file = os.path.join(local_dir, "latest.pt")
    replay_dir = os.path.join(local_dir, "replay")
    
    os.makedirs(replay_dir, exist_ok=True)

    client = ServerClient(args.server, args.worker)
    generator = SelfPlayGenerator()

    local_version = -1
    epoch = -1
    while True:
        epoch+=1
        print("=" * 60)

        #
        # 1. Download latest model if needed
        #
        server_info = client.get_version()
        
        if server_info is None:
            print("Could not connect to server or parse version. Retrying in 10 seconds...")
            time.sleep(10)
            continue

        server_version = int(server_info.get("last_updated_model", -1))
        server_phase = int(server_info.get("current_phase", 0))

        if server_version != local_version or generator.latest_model_path is None:
            print(f"Downloading model version {server_version}")
            try:
                client.download_latest_model(model_file)
                generator.load_model(model_file)
                local_version = server_version
            except Exception as e:
                print(f"Error downloading model: {e}. Retrying in 10 seconds...")
                time.sleep(10)
                continue

        #
        # 2. Generate games
        #
        try:
            replay_file = generator.play_games(
                num_games=args.games,
                save_dir=replay_dir,
                worker_id=args.worker,
                model_version=local_version,
                current_phase=server_phase,
                epoch=epoch
            )
        except Exception as e:
            print(f"Error during self-play generation: {e}")
            replay_file = None

        if not replay_file or not os.path.exists(replay_file):
            print("No replay file generated. Retrying...")
            time.sleep(5)
            continue

        #
        # 3. Upload replay
        #
        print(f"Uploading {replay_file} to server...")
        try:
            client.upload_replay(replay_file, model_version=local_version)
            
            # 4. Remove local replay only on successful upload
            os.remove(replay_file)
            print("Batch completed and uploaded successfully.")
            
        except Exception as e:
            print(f"Failed to upload replay: {e}. File kept locally for next batch.")
            time.sleep(10)

if __name__ == "__main__":
    main()