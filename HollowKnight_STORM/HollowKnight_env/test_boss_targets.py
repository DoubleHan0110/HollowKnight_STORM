import time
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HollowKnight_env.HKenv import HKEnv

def main():
    env = HKEnv()
    obs, info = env.reset()
    print(f"初始 targets: {sorted(env.boss_targets)} (共 {len(env.boss_targets)} 个)")

    prev = set(env.boss_targets)
    reward_sum = 0
    new_reward_sum = 0

    try:
        while True:
            # 空动作
            action = np.zeros(env.num_actions, dtype=np.int8)
            obs, reward, terminated, truncated, info = env.step(action)
            new_reward_sum += reward
            if new_reward_sum > reward_sum:
                reward_sum = new_reward_sum
                print(f"reward_sum: {reward_sum}")

            # cur = set(env.boss_targets)
            # if cur != prev:
            #     removed = prev - cur
            #     if removed:
            #         print(f"[Boss 击破] 移除: {sorted(list(removed))} | 剩余: {len(cur)} -> {sorted(list(cur))}")
            #     prev = cur

            if terminated or truncated:
                print(f"episode 结束, terminated={terminated}, truncated={truncated}")
                print(f"reward_sum: {reward_sum}")
                break

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()