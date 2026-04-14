#!/bin/bash
set -e

source /home/nyuair/anaconda3/etc/profile.d/conda.sh
conda activate kat
export COPPELIASIM_ROOT=~/zhewen/robo/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export OPENAI_API_KEY=${OPENAI_API_KEY:?Please set OPENAI_API_KEY}
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms
export QT_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:0
cd ~/zhewen/robo/KAT

RES=256

# reach_target: success n=5, failure n=1
echo "=== reach_target success ==="
python kat_baseline/scripts/record_episode.py --task reach_target --n_demos 5 --seed 1002 --resolution $RES 2>&1 | tail -3
echo "=== reach_target failure ==="
python kat_baseline/scripts/record_episode.py --task reach_target --n_demos 1 --seed 1001 --resolution $RES 2>&1 | tail -3

# push_button: success n=5, failure n=1
echo "=== push_button success ==="
python kat_baseline/scripts/record_episode.py --task push_button --n_demos 5 --seed 1002 --resolution $RES 2>&1 | tail -3
echo "=== push_button failure ==="
python kat_baseline/scripts/record_episode.py --task push_button --n_demos 1 --seed 1002 --resolution $RES 2>&1 | tail -3

# pick_up_cup: success n=5, failure n=1
echo "=== pick_up_cup success ==="
python kat_baseline/scripts/record_episode.py --task pick_up_cup --n_demos 5 --seed 1002 --resolution $RES 2>&1 | tail -3
echo "=== pick_up_cup failure ==="
python kat_baseline/scripts/record_episode.py --task pick_up_cup --n_demos 1 --seed 1002 --resolution $RES 2>&1 | tail -3

# take_lid_off_saucepan: success n=5, failure n=1
echo "=== take_lid_off_saucepan success ==="
python kat_baseline/scripts/record_episode.py --task take_lid_off_saucepan --n_demos 5 --seed 1002 --resolution $RES 2>&1 | tail -3
echo "=== take_lid_off_saucepan failure ==="
python kat_baseline/scripts/record_episode.py --task take_lid_off_saucepan --n_demos 1 --seed 1002 --resolution $RES 2>&1 | tail -3

# stack_blocks: failure only (no success possible)
echo "=== stack_blocks failure ==="
python kat_baseline/scripts/record_episode.py --task stack_blocks --n_demos 5 --seed 1002 --resolution $RES 2>&1 | tail -3

echo "=== ALL DONE ==="
