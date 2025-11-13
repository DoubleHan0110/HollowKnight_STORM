@echo off
set run_name=HollowKnight-life_done-wm_2L256D4H-100k-seed1_2
set num_episode=5

python eval_HK.py ^
    -run_name "%run_name%" ^
    -config_path "config_files/STORM.yaml" ^
    -num_episode %num_episode%

pause

