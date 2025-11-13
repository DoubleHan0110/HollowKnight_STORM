from fastapi import FastAPI
import uvicorn
from collections import deque
import time
from typing import List, Dict

app = FastAPI()

# 事件队列：存储打击和被打击事件
hit_events = deque(maxlen=100)  # 打击敌人事件
damage_events = deque(maxlen=100)  # 被打击事件

@app.get("/soul_gain/{amount}")
async def soul_gain(amount: int):
    # print(f"soul gain: {amount}")
    return "OK"

@app.get("/hit/{entity_name}/{damage}/{remaining_hp}")
async def hit(entity_name: str, damage: int, remaining_hp: int):
    hit_events.append({
        "entity": entity_name,
        "damage": damage,
        "remaining_hp": remaining_hp,
        "time": time.time()
    })
    print(f"hit {entity_name}, damage: {damage}, remaining hp: {remaining_hp}")
    return "OK"

@app.get("/take_hit/{hazard_type}/{damage}")
async def take_hit(hazard_type: str, damage: int):
    damage_events.append({
        "hazard_type": hazard_type,
        "damage": damage,
        "time": time.time()
    })
    print(f"got damaged")
    return "OK"

# 新增：获取自上次查询以来的新事件（用于 gym env 的 step）
@app.get("/get_events")
async def get_events(last_check_time: float = 0.0):
    """
    获取自 last_check_time 以来的新事件
    """
    current_time = time.time()
    
    # 筛选出新的打击事件
    new_hits = [
        event for event in hit_events 
        if event["time"] > last_check_time
    ]
    
    # 筛选出新的被打击事件
    new_damages = [
        event for event in damage_events 
        if event["time"] > last_check_time
    ]
    
    return {
        "hits": new_hits,
        "damages": new_damages,
        "current_time": current_time
    }

# 新增：清除所有事件（用于 reset）
@app.post("/clear_events")
async def clear_events():
    hit_events.clear()
    damage_events.clear()
    return "OK"

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=9393,
        log_level="warning",
        access_log=False,
        loop="asyncio"
    )