@echo off
set RUNS=runs/HollowKnight-life_done-wm_2L256D4H-100k-seed1_2, runs/HollowKnight-life_done-wm_2L256D4H-100k-seed1_3, runs/HollowKnight-life_done-wm_2L256D4H-100k-seed1_4
tensorboard --logdir "%RUNS%" --port 6006 --host localhost
pause