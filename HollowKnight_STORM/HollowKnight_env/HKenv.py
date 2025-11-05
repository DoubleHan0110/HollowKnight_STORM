import gymnasium as gym
import pygetwindow as gw
import dxcam
import time
import cv2
import keyboard

import os

class HKEnv(gym.Env):
    def __init__(self):
        # 初始化时就找到窗口和创建摄像头，避免重复创建
        self.window = None
        self.camera = None
        self._setup_windows()

    

    def reset(self):
        """重置环境，重新进入boss房间"""
        # 清理之前的状态
        self._cleanup_keys()
        
        # 重新进入boss房间
        self._enter_boss_room()
        
        # 获取初始观测
        obs = self._get_latest_frame()
        return obs  # gymnasium 格式
    


    def step(self, action):
        pass

    def _setup_windows(self):
        """初始化窗口和摄像头"""
        # 查找空洞骑士窗口
        windows = gw.getWindowsWithTitle("Hollow Knight")
        self.window = windows[0]

        self.camera = dxcam.create(output_idx=0)
    
    def _get_latest_frame(self):
        """获取最新帧"""

        self.window.activate()
        
        # 计算截屏区域
        region = (self.window.left, self.window.top, self.window.left + self.window.width, self.window.top + self.window.height)
        
        frame = self.camera.grab(region=region)                    
        return frame

    def _enter_boss_room(self):
        """重新进入boss房间"""
        print("正在重新进入boss房间...")
        time.sleep(1.0)
        
        # 1. 等待并寻找菜单界面
        max_attempts = 10
        for attempt in range(max_attempts):
            # 尝试按w键导航到正确位置
            keyboard.send('w')
            time.sleep(1.0)
            
            # 检查是否到达了挑战界面
            if self.is_challenge_menu():
                print("✅ 找到挑战菜单")
                break
        else:
            print("未能找到挑战菜单，继续尝试...")
        
        # 2. 按空格进入boss房
        keyboard.send('space')
        print("按下空格键进入挑战...")
        
        # 3. 等待加载完成
        # self._wait_for_loading()
        print("成功进入boss房间")
    
    def is_challenge_menu(self):
       pass
    
    def _wait_for_loading(self):
        """等待游戏加载完成"""
        print("等待加载完成...")
        ready = False
        
        while True:
            frame = self._get_latest_frame()
            if frame is None:
                continue
                
            # 检测是否在加载界面（通常是黑屏或很暗）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            is_loading = (gray < 20).sum() < 100  # 调整阈值
            
            if ready and not is_loading:
                break
            else:
                ready = is_loading
            
            time.sleep(0.1)
        
        # 额外等待确保完全准备好
        time.sleep(2.0)

    def _cleanup_keys(self):
        """清理可能按住的按键"""
        keys_to_release = ['a', 'd', 'w', 's', 'j', 'k', 'space']
        for key in keys_to_release:
            keyboard.release(key)
    
    def _cleanup(self):
        """清理camera"""
        if self.camera:
            try:
                self.camera.stop()
            except:
                pass
            self.camera = None

