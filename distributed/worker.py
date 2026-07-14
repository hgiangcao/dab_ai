import os
import shutil
import time

from client import ServerClient
from selfplay import SelfPlayGenerator


SERVER = "172.16.2.31:8000"

WORKER_NAME = "worker01"

LOCAL_DIR = "./worker_data"

MODEL_FILE = os.path.join(LOCAL_DIR, "latest.pt")
REPLAY_DIR = os.path.join(LOCAL_DIR, "replay")

GAMES_PER_BATCH = 2


def main():
    os.makedirs(REPLAY_DIR, exist_ok=True)

    client = ServerClient(SERVER, WORKER_NAME)
    generator = SelfPlayGenerator()

    local_version = -1

    while True:
        print("=" * 60)

        #
        # 1. Download latest model if needed
        #
        server_version = client.get_version()
        
        if server_version == -1:
            print("Could not connect to server or parse version. Retrying in 10 seconds...")
            time.sleep(10)
            continue

        if server_version != local_version or generator.latest_model_path is None:
            print(f"Downloading model version {server_version}")
            try:
                client.download_latest_model(MODEL_FILE)
                generator.load_model(MODEL_FILE)
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
                num_games=GAMES_PER_BATCH,
                save_dir=REPLAY_DIR,
                worker_id=WORKER_NAME,
                model_version=local_version,
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