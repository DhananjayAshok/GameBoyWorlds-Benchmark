#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source configs/config.env
setup/.venv/bin/python runs/contamination/answer_openrouter.py --model google/gemini-3.1-pro-preview
