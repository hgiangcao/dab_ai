import requests


class ServerClient:

    def __init__(self, server, worker):
        if not server.startswith("http://") and not server.startswith("https://"):
            server = "http://" + server
        self.server = server
        self.worker = worker

    def get_version(self):

        try:
            r = requests.get(self.server + "/version", timeout=10)
            data = r.json()
            print (data)
            # Our server returns {"run": ..., "last_updated_model": X, "current_phase": Y}
            return int(data.get("last_updated_model", -1))
        except Exception as e:
            print(f"Connection error fetching version: {e}")
            return -1

    def download_latest_model(self, save_path):

        r = requests.get(
            self.server + "/latest_model",
            stream=True
        )

        with open(save_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

    def upload_replay(self, filename, model_version=0):
        with open(filename, "rb") as f:
            requests.post(
                self.server + "/upload_replay",
                params={
                    "worker": self.worker,
                    "model_version": model_version,
                },
                files={
                    "file": f
                }
            )