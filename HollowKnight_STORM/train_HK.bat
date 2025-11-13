@echo off
setlocal

:: 设置环境变量
set env_name=HollowKnight

:: 运行 HollowKnight 专用训练脚本
python -u train_HK.py ^
    -n "%env_name%-life_done-wm_2L256D4H-100k-seed1_3" ^
    -seed 1 ^
    -config_path "config_files/STORM_HK.yaml" ^
    -trajectory_path "D_TRAJ/%env_name%.pkl"

endlocal
pause

