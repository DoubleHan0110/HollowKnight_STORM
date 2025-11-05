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

        # 菜单阈值
        self.menu_threshold = 0.4
        self.white_pixel_threshold = 500000
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
        self.window.activate()
    
    def _get_latest_frame(self):
        """获取最新帧"""

        # self.window.activate()

        screen_width = self.camera.width
        screen_height = self.camera.height
        left = max(0, self.window.left)
        top = max(0, self.window.top)
        right = min(screen_width, self.window.left + self.window.width)
        bottom = min(screen_height, self.window.top + self.window.height)
        
        region = (left, top, right, bottom)
        
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
            frame = self._get_latest_frame()
            if self._is_challenge_menu(frame):
                print("✅ 找到挑战菜单")
                break
            else:
                print("未能找到挑战菜单，继续尝试...")
        
        # 2. 按空格进入boss房
        keyboard.send('space')
        print("按下空格键进入挑战...")
        time.sleep(1.0)
        print("进入wait_for_loading时间")
        
        # 3. 等待加载完成
        self._wait_for_loading()
        print("成功进入boss房间")
    
    def _is_challenge_menu(self, frame, visualize=False):


        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 可以只截取右上角区域来提速（菜单位置固定）
        h, w = gray.shape
        roi = gray[int(h * 0.05):int(h * 0.85), int(w * 0.40):w]

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        TEMPLATE_PATH = os.path.join(BASE_DIR, "locate_fig", "menu_icon.png")
        template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 匹配判断
        res = cv2.matchTemplate(roi, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        # print(f"max_val: {max_val}")

        # 调试用
        if visualize:
            cv2.imshow("template_gray", template_gray)
            # 显示ROI区域
            cv2.imshow("ROI区域 (右上角)", roi)
            cv2.waitKey(20000)

        return max_val > self.menu_threshold
       
    
    def _wait_for_loading(self):
        """等待游戏加载完成"""
        print("等待加载完成...")
        ready = False
        
        while True:
            frame = self._get_latest_frame()
            if frame is None:
                continue
                
            # 检测是否在加载界面
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            # print(f"num of gray > 250: {(gray > 250).sum()}")

            # 判断白屏像素数量
            is_loading = (gray > 250).sum() > self.white_pixel_threshold  
            
            if ready and not is_loading:
                break
            else:
                ready = is_loading
            
            time.sleep(0.1)
        
        # 额外等待确保完全准备好
        time.sleep(2.0)
        print("加载完成")

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
    
    def test_locate_menu(self):
        frame = self._get_latest_frame()
        value = self._is_challenge_menu(frame, visualize=False)
        if value:
            print("✅ 找到挑战菜单")
        else:
            print("❌ 未找到挑战菜单", value)

if __name__ == "__main__":
    env = HKEnv()
    env.reset()

    # keyboard.send('space')
