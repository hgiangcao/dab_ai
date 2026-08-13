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
    best_model_file = os.path.join(local_dir, "best.pt")
    replay_dir = os.path.join(local_dir, "replay")
    
    os.makedirs(replay_dir, exist_ok=True)

    client = ServerClient(args.server, args.worker)
    generator = SelfPlayGenerator()

    # Track locally which versions are already loaded so we skip redundant downloads
    local_version = -1          # tracks last_updated_model  (for 'self')
    local_best_checkpoint = -1  # tracks best_model_checkpoint (for 'best')
    epoch = -1
    while True:
        epoch += 1
        print("=" * 60)

        #
        # 1. Fetch server version info
        #
        server_info = client.get_version()

        if server_info is None:
            print("Could not connect to server or parse version. Retrying in 10 seconds...")
            time.sleep(10)
            continue

        server_version = int(server_info.get("last_updated_model", -1))
        server_phase = int(server_info.get("current_phase", 0))
        finish_pretrain = server_info.get("finish_pretrain", "False")
        server_best_checkpoint = int(server_info.get("best_model_checkpoint", 0))

        if finish_pretrain.lower() != "true":
            print("Server indicates pretraining is not finished. Waiting...")
            time.sleep(10)
            continue

        #
        # 2. Download latest model ('self') only when version changed
        #
        if server_version != local_version or generator.latest_model_path is None:
            print(f"Downloading model version {server_version} (self)")
            try:
                client.download_latest_model(model_file)
                generator.load_model(model_file)
                local_version = server_version
            except Exception as e:
                print(f"Error downloading latest model: {e}. Retrying in 10 seconds...")
                time.sleep(10)
                continue
        else:
            print(f"Self model already up-to-date (v{local_version}), skipping download.")

        #
        # 3. Download best model ('best') only when best_model_checkpoint changed
        #
        if server_best_checkpoint != local_best_checkpoint or generator.best_model_path is None:
            print(f"Downloading best model checkpoint v{server_best_checkpoint}")
            try:
                client.download_best_model(best_model_file)
                generator.load_best_model(best_model_file)
                local_best_checkpoint = server_best_checkpoint
            except Exception as e:
                print(f"Error downloading best model: {e}. Best opponent will fall back to self.")
                # Don't abort the epoch — best will gracefully fall back to self in selfplay.py
        else:
            print(f"Best model already up-to-date (checkpoint v{local_best_checkpoint}), skipping download.")

        #
        # 4. Generate games
        #
        try:
            replay_file = generator.play_games(
                num_games=args.games,
                save_dir=replay_dir,
                worker_id=args.worker,
                model_version=local_version,
                current_phase=server_phase
            )
        except Exception as e:
            print(f"Error during self-play generation: {e}")
            replay_file = None

        if not replay_file or not os.path.exists(replay_file):
            print("No replay file generated. Retrying...")
            time.sleep(5)
            continue

        #
        # 5. Upload replay
        #
        print(f"Uploading {replay_file} to server...")
        try:
            client.upload_replay(replay_file, model_version=local_version)

            # 6. Remove local replay only on successful upload
            os.remove(replay_file)
            print("Batch completed and uploaded successfully.")

        except Exception as e:
            print(f"Failed to upload replay: {e}. File kept locally for next batch.")
            time.sleep(10)

if __name__ == "__main__":
    main()