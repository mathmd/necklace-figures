#!/bin/sh
# Compute Figures 2-5 of the paper and write them to fig/.
set -e
cd "$(dirname "$0")"
python3 figure2.py
python3 figure3.py
python3 figures4_5.py
