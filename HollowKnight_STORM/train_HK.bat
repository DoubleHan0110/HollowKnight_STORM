@echo off
setlocal

:: 设置环境变量
set env_name=HollowKnight

:: 运行 HollowKnight 专用训练脚本
python -u train_HK.py ^
    -n "%env_name%_2L256D4H_150k_seed1_9FPS_bs64_twls0.5" ^
    -seed 1 ^
    -config_path "config_files/STORM_HK.yaml" ^
    -trajectory_path "D_TRAJ/%env_name%.pkl"

endlocal
pause

