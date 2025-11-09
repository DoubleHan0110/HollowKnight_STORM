# game_event_client.py
import requests
import threading
import time
from typing import List, Dict, Optional
from collections import deque
from fastapi import FastAPI
import uvicorn

class ModEventClient:
    """
    游戏事件客户端，用于从 FastAPI 服务器获取游戏事件
    """
    
    def __init__(self, base_url: str = "http://localhost:9393", last_check_time: float = 0.0):
        """
        初始化游戏事件客户端
        
        :param base_url: FastAPI 服务器地址
        :param start_server: 是否启动 FastAPI 服务器（如果为 True，会在后台线程启动）
        """
        self.base_url = base_url
        self.last_check_time = last_check_time
        self.server_thread = None
        
        self._start_server()
    
    def _start_server(self):
        """在后台线程启动 FastAPI 服务器"""
        app = FastAPI()
        
        # 事件队列
        hit_events = deque(maxlen=500)
        damage_events = deque(maxlen=500)
        
        @app.get("/soul_gain/{amount}")
        async def soul_gain(amount: int):
            return "OK"
        
        @app.get("/hit/{entity_name}/{damage}")
        async def hit(entity_name: str, damage: int):
            hit_events.append({
                "entity": entity_name,
                "damage": damage,
                "time": time.time()
            })
            return "OK"
        
        @app.get("/take_hit/{hazard_type}/{damage}")
        async def take_hit(hazard_type: str, damage: int):
            damage_events.append({
                "hazard_type": hazard_type,
                "damage": damage,
                "time": time.time()
            })
            return "OK"
        
        @app.get("/get_events")
        async def get_events(last_check_time: float = 0.0, end_time: float | None = None):
            now = time.time()
            hi = end_time if end_time is not None else now
            new_hits = [e for e in hit_events if (e["time"] > last_check_time and e["time"] <= hi)]
            new_damages = [e for e in damage_events if (e["time"] > last_check_time and e["time"] <= hi)]

            return {
                "hits": new_hits,
                "damages": new_damages,
                "current_time": now
            }

        
        @app.post("/clear_events")
        async def clear_events():
            hit_events.clear()
            damage_events.clear()
            return "OK"
        
        # 在后台线程启动服务器
        def run_server():
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=9393,
                log_level="warning",
                access_log=False,
                loop="asyncio"
            )
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(1)  # 等待服务器启动
    
    def get_events_since_last_check(self, current_time: float = None) -> Dict:
        """
        获取自上次检查以来的新事件
        
        :return: 包含 hits 和 damages 的字典
        """
        query_time = self.last_check_time
        try:
            response = requests.get(
                f"{self.base_url}/get_events",
                params={"last_check_time": query_time, "end_time": current_time},
                timeout=0.1
            )
            if response.status_code == 200:
                data = response.json()
                self.last_check_time = current_time
                return data
        except Exception as e:
            print(f"获取事件失败: {e}")
        
        return {"hits": [], "damages": [], "current_time": time.time()}
    
    def reset(self, last_check_time: float = None):
        """重置事件检查时间（用于 reset）"""
        if last_check_time is not None:
            self.last_check_time = last_check_time
        else:
            self.last_check_time = time.time()

        self._clear_events()
    
    def _clear_events(self):
        """清除所有事件（用于 reset）"""
        try:
            requests.post(f"{self.base_url}/clear_events", timeout=0.1)
        except Exception as e:
            print(f"清除事件失败: {e}")
    
    

if __name__ == "__main__":
    mod_event_client = ModEventClient()
    mod_event_client.reset()
    print("等待游戏事件...（请确保游戏 Mod 正在运行）")
    
    while True:
        time.sleep(0.1)  # 每 0.1 秒检查一次
        events = mod_event_client.get_events_since_last_check()
        
        # 检查是否有实际事件（不是空列表）
        if events["hits"] or events["damages"]:
            print(f"收到事件: {events}")
        elif len(events.get("hits", [])) == 0 and len(events.get("damages", [])) == 0:
            # 可以添加一个计数器，避免一直打印空消息
            pass

