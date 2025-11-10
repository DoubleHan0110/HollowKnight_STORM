@echo off
setlocal

set env_name=HollowKnight

python -u train.py ^
    -n "%env_name%-life_done-wm_2L512D8H-100k-seed1" ^
    -seed 1 ^
    -config_path "config_files/STORM.yaml" ^
    -env_name "%env_name%" ^
    -trajectory_path "D_TRAJ/%env_name%.pkl"

endlocal
pause
