#提供接口，实现 给定位置坐标 → 返回动作ID (0上/1下/2左/3右)

"""
模型推理接口 - 供真机集成框架调用
功能：给定位置坐标 → 返回动作ID (0上/1下/2左/3右)
适配 8 维观测空间（包含四个方向的可走性）
示例：
    from rl_ppo_module.model_interface import get_action_from_position
    
    action = get_action_from_position(6, 4)
    # action 返回 0~3，对应 上/下/左/右
"""
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径（兼容从任何位置导入）
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from env_module.GridEnv import GridNavEnv

# ========== 全局变量（模块级单例，只加载一次） ==========
_MODEL = None
_ENV = None
_GOAL_POS = None


def _ensure_loaded():
    global _MODEL, _ENV, _GOAL_POS
    if _MODEL is None:
        # 1. 创建环境实例（获取终点坐标和障碍物信息）
        _ENV = GridNavEnv()
        _ENV.reset()
        _GOAL_POS = _ENV.goal_pos
        
        # 2. 加载训练好的模型
        model_path = Path(__file__).parent.parent / "models" / "ppo_grid_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        _MODEL = PPO.load(str(model_path))
        print(f"[model_interface] ✅ 模型加载成功: {model_path}")
        print(f"[model_interface] 目标点固定为: {_GOAL_POS}")
    return _MODEL, _ENV, _GOAL_POS


def get_action_from_position(x: int, y: int) -> int:
    """
    给定当前位置坐标，返回动作ID
    
    参数：
        x: 列号 (0 ~ 15)
        y: 行号 (0 ~ 20)
    
    返回：
        int: 0=上, 1=下, 2=左, 3=右
    """
    model, env, goal = _ensure_loaded()
    
    # 构建 8 维观测值: [x, y, goal_x, goal_y, up, down, left, right]
    gx, gy = goal
    obstacles = set(env.obstacles)
    
    # ===== 修正：up/down 与 GridEnv.py 保持一致 =====
    # 坐标系：左下角原点，y 向上增加
    # up  = y+1 方向（上），down = y-1 方向（下）
    up = 0 if (y + 1 >= env.grid_size_y or (x, y + 1) in obstacles) else 1
    down = 0 if (y - 1 < 0 or (x, y - 1) in obstacles) else 1
    left = 0 if (x - 1 < 0 or (x - 1, y) in obstacles) else 1
    right = 0 if (x + 1 >= env.grid_size_x or (x + 1, y) in obstacles) else 1
    
    obs = np.array([x, y, gx, gy, up, down, left, right], dtype=np.float32)
    
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


# ========== 快速测试入口 ==========
if __name__ == "__main__":
    test_points = [
                    (1, 1), (5, 5), (10, 10), (1, 7), (14, 2), (14, 13), (15,17)
                   ]
    print("=" * 50)
    print("接口测试（坐标 → 动作）")
    for x, y in test_points:
        action = get_action_from_position(x, y)
        action_names = {0: "↑上", 1: "↓下", 2: "←左", 3: "→右"}
        print(f"  位置 ({x:2d}, {y:2d}) → 动作 {action} ({action_names[action]})")
    print("=" * 50)