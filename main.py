import os
import subprocess
import sys
import time
from configs.config import *
PYTHON = sys.executable

DATA_PATH = os.path.dirname(os.path.abspath(__file__))
training = False         # Set True if you want to train on your pc.


def run_training():
    print("\nUse config.py to control Training models....\n")
    result = subprocess.run([PYTHON, "scripts/train.py"], cwd=DATA_PATH)
    if result.returncode != 0:
        print("Training failed. Fix the error above before continuing.")
        sys.exit(1)


def start_api():
    print("\nStarting FastAPI server on http://localhost:8000 ...\n")
    return subprocess.Popen([PYTHON, "-m", "uvicorn", "scripts.api:app", "--port", "8000"], cwd=DATA_PATH)


def start_ui():
    print("\nStarting Streamlit UI on http://localhost:8501 ...\n")
    subprocess.run([PYTHON, "-m", "streamlit", "run", "scripts/streamlit_app.py"], cwd=DATA_PATH)


if __name__ == "__main__":
    if training:
        run_training()
    api_process = start_api()
    time.sleep(10)
    try:
        start_ui()
    finally:
        api_process.terminate()


'''
cd brain_tumor_segmentation
uvicorn scripts.api:app --port 8000
streamlit run scripts/streamlit_app.py
'''
