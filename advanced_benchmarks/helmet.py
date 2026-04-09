# -*- coding: utf-8 -*-
"""
HELMET

Problem: Existing long-context benchmarks provided noisy signals due to limited
application coverage, insufficient lengths, unreliable metrics, and
incompatibility with base models.

Setup (run manually):
    git clone https://github.com/princeton-nlp/HELMET.git
    cd HELMET
    pip install -r requirements.txt
    bash scripts/download_data.sh
"""

import subprocess
import os


def setup_helmet(target_dir="HELMET"):
    """Clone and set up the HELMET benchmark."""
    if not os.path.exists(target_dir):
        subprocess.run(["git", "clone", "https://github.com/princeton-nlp/HELMET.git", target_dir], check=True)

    req_file = os.path.join(target_dir, "requirements.txt")
    if os.path.exists(req_file):
        subprocess.run(["pip", "install", "-q", "-r", req_file], check=True)

    download_script = os.path.join(target_dir, "scripts", "download_data.sh")
    if os.path.exists(download_script):
        subprocess.run(["bash", download_script], cwd=target_dir, check=True)


if __name__ == "__main__":
    setup_helmet()
    print("HELMET setup complete. Check HELMET/data/ for downloaded benchmarks.")
