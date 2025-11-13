@echo off
echo Starting TensorBoard...
echo Open your browser and go to: http://localhost:6006
echo.
tensorboard --logdir runs/HollowKnight-life_done-wm_2L256D4H-100k-seed1 --port 6006 --host localhost --reload_interval 1
pause