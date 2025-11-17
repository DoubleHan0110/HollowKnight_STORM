import gymnasium as gym
import pygetwindow as gw
import dxcam
import time
import cv2
import keyboard
import sys
import os
import numpy as np
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from .mod_event_class import ModEventClient
from env_wrapper import LifeLossInfo

class HKEnv(gym.Env):
    def __init__(self):
        
        self.window = None
        self.camera = None

        self.process_time = 0.0

        # 菜单阈值
        self.menu_threshold = 0.99
        self.white_pixel_threshold = 500000

        self.gap = 1.0 / 9.0
        self._prev_time = None

        self.KEYMAP = {
            0: ('a', 'move_left', 'hold'),      # 按住左
            1: ('d', 'move_right', 'hold'),     # 按住右
            2: ('w', 'look_up', 'hold'),        # 按住上（向上看）
            3: ('s', 'look_down', 'hold'),     # 按住下（向下看）
            4: ('j', 'attack', 'instant'),     # 攻击（瞬间）
            5: ('k', 'dash', 'instant'),        # 冲刺（瞬间）
            6: ('space', 'jump', 'hold'),       # 跳跃（按住）
        }
        self.action_keys = [self.KEYMAP[i][0] for i in range(len(self.KEYMAP))]
        self.num_actions = len(self.action_keys)
        self.instant_keys = {4, 5}
        self._key_states = np.zeros(self.num_actions, dtype=np.int8)

        # 定义动作空间
        self.action_space = gym.spaces.MultiBinary(self.num_actions)

        self.lives_info = None

        self.boss_targets = None
        self._episode_frame_number = 0

        self._setup_windows()

        # 定义观测空间
        H, W = self.window.height, self.window.width
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8)

        self.mod_event_client = ModEventClient()

    def reset(self, seed=None, options=None):
        """重置环境，重新进入boss房间"""
        super().reset(seed=seed)

        self._cleanup_keys()
        
        # 重新进入boss房间
        self._enter_boss_room()

        self._prev_time = time.time()

        self.mod_event_client.reset(last_check_time=self._prev_time)
        
        obs = self._get_latest_frame()
        
        self.boss_targets = {"Mantis Lord", "Mantis Lord S1", "Mantis Lord S2", "Mantis Lord S3"}
        self.lives_info = 9
        self._episode_frame_number = 0
        info = {}
        info["lives"] = self.lives_info
        info["episode_frame_number"] = self._episode_frame_number
        return obs, info  
    


    def step(self, action):

        self._execute_actions(action)
        # 控制采样频率
        if self._prev_time is not None:
            elapsed = time.time() - self._prev_time
            sleep_time = self.gap - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        # process_gap = time.time() - self.process_time
        # print(f"process_gap: {process_gap}")

        self._prev_time = time.time()

        # self.process_time = time.time()   

        obs = self._get_latest_frame()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        current_time = self._prev_time
        # print(f"current_time: {current_time}")
        events = self.mod_event_client.get_events_since_last_check(current_time=current_time)


        if events["hits"]:
            # reward += 1
            # print(f"length of hits: {len(events['hits'])}")
            for ev in events["hits"]:
                name = ev.get("entity", "")
                hp = ev.get("remaining_hp", 0)
                reward += 1
                if name in self.boss_targets and hp <= 0:
                    self.boss_targets.remove(name)
                    # print(f"[Mantis] defeated: {name}, remaining targets: {len(self.boss_targets)}")
            # print(f"hit the boss, reward: {reward}")    

        if events["damages"]:
            # print(f"damages: {events['damages']}")
            total_damage = sum(event["damage"] for event in events["damages"])
            self.lives_info -= total_damage
            if self.lives_info < 0:
                self.lives_info = 0
            # print(f"got damaged, damage this time: {total_damage}, lives: {self.lives_info}")
            
        info["lives"] = self.lives_info
        self._episode_frame_number += 1
        info["episode_frame_number"] = self._episode_frame_number

        if obs is None:
            obs = np.zeros((self.window.height, self.window.width, 3), dtype=np.uint8)
            truncated = True
        
        if self.lives_info == 0 or len(self.boss_targets) == 0:
            terminated = True
        
        if terminated or truncated:
            self._cleanup_keys()
            self._wait_for_loading()

        return obs, reward, terminated, truncated, info

    def _setup_windows(self):
        """初始化窗口和camera"""
        # 查找空洞骑士窗口
        windows = gw.getWindowsWithTitle("Hollow Knight")
        self.window = windows[0]
        self.window.activate()
        time.sleep(0.1)

        self.camera = dxcam.create(output_idx=0)
        
    
    def _get_latest_frame(self):
        """获取最新帧"""

        screen_width = self.camera.width
        screen_height = self.camera.height
        left = max(0, self.window.left)
        top = max(0, self.window.top)
        right = min(screen_width, self.window.left + self.window.width)
        bottom = min(screen_height, self.window.top + self.window.height)
        
        region = (left, top, right, bottom)

        for attempt in range(5):
            frame = self.camera.grab(region=region)
            if frame is not None:
                break
            time.sleep(0.2)
            self.window.activate()
        
        if frame is None:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def _enter_boss_room(self):
        """重新进入boss房间"""
        # print("正在重新进入boss房间...")
        time.sleep(1.0)
        
        max_attempts = 10
        found_menu = False
        for attempt in range(max_attempts):
            keyboard.send('w')
            time.sleep(2.0)
            
            # 查看是否达到挑战菜单
            for attempt in range(2):
                frame = self._get_latest_frame()
                if frame is not None and self._is_challenge_menu(frame):
                    # print("找到挑战菜单")
                    found_menu = True
                    break
                time.sleep(0.5)
            if found_menu:
                break
        
        # 进入boss房
        keyboard.send('space')
        
        time.sleep(1.0)
        # print("进入wait_for_loading")
        
        # 等待加载完成
        self._wait_for_loading()
        # print("成功进入boss房间")
    
    def _is_challenge_menu(self, frame, visualize=False):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 只截取右上角区域
        h, w = gray.shape
        roi = gray[int(h * 0.05):int(h * 0.85), int(w * 0.40):w]

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        TEMPLATE_PATH = os.path.join(BASE_DIR, "locate_fig", "menu_icon2.png")
        template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(roi, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        # print(f"max_val: {max_val}")

        # 调试用
        if visualize:
            cv2.imshow("template_gray", template_gray)
            cv2.imshow("ROI区域 (右上角)", roi)
            cv2.waitKey(20000)

        return max_val > self.menu_threshold
       
    
    def _wait_for_loading(self):
        """等待游戏加载完成"""
        # print("等待加载...")
        ready = False
        is_loading = False
        
        while True:
            frame = self._get_latest_frame()
            if frame is None:
                print("报错：当前截屏失败，等待空格键继续：")
                keyboard.wait('space')

                break
            else:
                # 检测是否在加载界面
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        # print("加载完成")

    def _execute_actions(self, action):
        """执行动作"""
        action = np.array(action, dtype=np.int8).flatten()

        for i in range(self.num_actions):
            key_name, _, _ = self.KEYMAP[i]
            new_state = action[i]
            old_state = self._key_states[i]
            
            if i in self.instant_keys:
                if new_state == 1:
                    keyboard.send(key_name)
            else:
                if new_state == 1 and old_state == 0:
                    keyboard.press(key_name)
                elif new_state == 0 and old_state == 1:
                    keyboard.release(key_name)

        self._key_states = action.copy()

    def _cleanup_keys(self):
        """清理可能按住的按键"""
        keys_to_release = ['a', 'd', 'w', 's', 'j', 'k', 'space']
        for key in keys_to_release:
            keyboard.release(key)

        self._key_states = np.zeros(self.num_actions, dtype=np.int8)
    
    def _cleanup(self):
        """清理camera"""
        if self.camera:
            try:
                self.camera.stop()
            except:
                pass
            self.camera = None
    


if __name__ == "__main__":
    env = HKEnv()
    # env = gym.wrappers.ResizeObservation(env, shape = (64, 64))
    # env = LifeLossInfo(env)
    # episode =2
    # for i in range(episode):
    #     obs, info = env.reset()
    #     while True:
    #         action = env.action_space.sample()
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         if terminated:
    #             print(f"episode {i} terminated")
    #             break
    #         if truncated:
    #             break
    #     print(f"episode {i} finished")
    # env.reset()
    env._is_challenge_menu(env._get_latest_frame())
