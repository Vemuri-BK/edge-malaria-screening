#!/usr/bin/env bash
# One-time Raspberry Pi setup for the EdAi malaria project.
# Run ON THE PI from the project folder:   bash pi/setup_pi.sh
set -e
echo "== 1/4 System packages (git, Python venv, compiler for llama.cpp) =="
sudo apt update
sudo apt install -y git python3-venv python3-pip build-essential cmake python3-gpiozero python3-lgpio

echo "== 2/4 Python environment (.venv, can also see the system GPIO libraries) =="
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip

echo "== 3/4 CNN inference packages (no PyTorch needed) =="
pip install -r pi/requirements.txt

echo "== 4/4 LLM explainer (compiles llama.cpp for the Pi's ARM CPU: ~15-30 min) =="
pip install llama-cpp-python huggingface_hub

mkdir -p outputs models results
echo
echo "Done. Each new SSH session: cd ~/edge-malaria-screening && source .venv/bin/activate"
echo "Test the LED:  python pi/led_test.py"
