@echo off
set env_name=HollowKnight
python eval.py ^
    -env_name "%env_name%" ^
    -run_name "%env_name%-life_done-wm_2L256D4H-100k-seed1_2" ^
    -config_path "config_files/STORM.yaml"
pause

