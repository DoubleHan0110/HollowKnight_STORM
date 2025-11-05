import dxcam
import cv2
import time
import pygetwindow as gw


import numpy as np
from HKenv import HKEnv
def get_hollow_knight_frame():
    """获取空洞骑士游戏窗口的当前帧"""
    
    # 1. 查找空洞骑士窗口
    print("正在查找空洞骑士窗口...")
    windows = gw.getWindowsWithTitle("Hollow Knight")
    
    if not windows:
        print("❌ 未找到空洞骑士窗口！请确保游戏已启动")
        return None
    
    window = windows[0]
    print(f"✅ 找到窗口: {window.title}")
    print(f"   窗口位置: ({window.left}, {window.top})")
    print(f"   窗口大小: {window.width} x {window.height}")
    
    # 2. 激活窗口到前台
    try:
        window.activate()
        # print("✅ 窗口已激活")
    except:
        print("⚠️  窗口激活失败，但继续尝试截屏")
    
    # 3. 创建截屏器
    print("正在初始化截屏器...")
    camera = dxcam.create()
    
    # 4. 设置截屏区域（窗口范围），确保不超出屏幕边界
    screen_width = camera.width
    screen_height = camera.height
    # print(f"屏幕分辨率: {screen_width} x {screen_height}")
    
    # 计算截屏区域，限制在屏幕范围内
    left = max(0, window.left)
    top = max(0, window.top)
    right = min(screen_width, window.left + window.width)
    bottom = min(screen_height, window.top + window.height)
    
    region = (left, top, right, bottom)
    print(f"截屏区域: {region}")
    
    # 5. 截取一帧
    print("正在截取游戏画面...")
    frame = camera.grab(region=region)
    
    if frame is None:
        print("❌ 截屏失败！")
        return None
    
    print(f"✅ 成功截取画面，尺寸: {frame.shape}")
    return frame

def test_frame_capture():
    """测试帧捕获功能"""
    print("=" * 50)
    print("空洞骑士帧捕获测试")
    print("=" * 50)
    
    # 获取一帧
    frame = get_hollow_knight_frame()
    
    if frame is not None:
        # 显示帧信息
        print(f"帧数据类型: {type(frame)}")
        print(f"帧形状: {frame.shape}")
        print(f"数据范围: {frame.min()} ~ {frame.max()}")
        
        # # 保存图片验证
        # cv2.imwrite("hollow_knight_frame.png", frame)
        # print("✅ 已保存截图到 hollow_knight_frame.png")
        
        # 显示图片（按 ESC 关闭）
        cv2.imshow("Hollow Knight Frame", frame)
        print("📷 图片已显示，按 ESC 键关闭窗口")
        
        # 等待按键
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                break
        
        cv2.destroyAllWindows()
        print("✅ 测试完成！")
    else:
        print("❌ 测试失败！")

if __name__ == "__main__":
    # test_frame_capture()
    env = HKEnv()
    value = env.is_challenge_menu()
    print(value)
    # print(frame.shape)
    # cv2.imshow("HK Frame", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # time.sleep(1)