#!/usr/bin/env bash
# Run once after extracting: bash INSTALL.sh
set -e
cd "$(dirname "$0")"
chmod +x run.sh selftest.py run_trials.py gates/*.py 2>/dev/null || true
echo "permissions set"
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
python3 -c 'import numpy,scipy' 2>/dev/null || pip install numpy scipy
( cd workload && make -s ) && echo "rategen built"
echo
echo "next:  ./run.sh check      (verify the machine)"
echo "       ./run.sh selftest   (36 correctness checks)"
