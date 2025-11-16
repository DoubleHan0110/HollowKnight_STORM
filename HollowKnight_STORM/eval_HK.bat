@echo off
set run_name=HollowKnight_2L256D4H_150k_seed1_9FPS_bs64_twls2
set num_episode=5

python eval_HK.py ^
    -run_name "%run_name%" ^
    -config_path "config_files/STORM_HK.yaml" ^
    -num_episode %num_episode%

pause

