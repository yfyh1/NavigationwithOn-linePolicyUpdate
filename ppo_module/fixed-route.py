#真机部署 通过输入给model_interface.py的get_action_from_position(x,y)函数，获取动作ID(0上/1下/2左/3右)

"""
fixed_route2.py
=================
执行层（真机 UDP 控制） + 主循环

职责：
  1. 交互式读取起点坐标
  2. 每一步问 model_interface.get_action_from_position(x, y) 要决策
  3. 把网格动作 0/1/2/3 翻译成机器狗 UDP 速度指令
  4. 追踪并打印"当前在第几格"，到终点自动停机

约定：
  - 机器狗初始朝向 = 网格的 +y 方向（"上"）
  - 1 个网格 = 0.6 米
  - 动作含义：0=上, 1=下, 2=左, 3=右

修复说明（相对原版 fixed_route2.py）：
  原 move_forward/backward/left/right 中
      step_count = int(total_distance / step_dist)
  int() 向下截断：0.6 / 0.243 = 2.469 → 2，每格只走 2×0.243=0.486m，
  每格累计少走 0.114m，走 17 格物理上差约 1.94m，造成严重位置漂移。

  修复为：步数向上取整（ceil），再把 total_distance 均匀分摊到每一步，
  使 step_count × actual_step == total_distance 严格成立，每格零截断误差。
  0.6m 现在会被拆成 3×0.2m 小步，走完正好 0.6m。
"""

import sys
import json
import math
import time
import socket
from pathlib import Path

# ---- 让 fixed_route2.py 能 import 同目录的 model_interface ----
sys.path.append(str(Path(__file__).resolve().parent))
# ---- 让 model_interface 能 import env_module ----
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model_interface import get_action_from_position, _ensure_loaded


# ============================================================
# 1. UDP 通信（原样保留）
# ============================================================
RELAY_IP = "192.168.123.220"
RELAY_PORT = 5091
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_vel(vx: float, vy: float, yaw: float):
    """发送三轴速度指令：前后vx / 左右vy / 旋转yaw"""
    cmd = {"vx": vx, "vy": vy, "yaw": yaw}
    send_data = json.dumps(cmd).encode("utf-8")
    udp_sock.sendto(send_data, (RELAY_IP, RELAY_PORT))
    print(f"    发送指令 vx={vx:.2f}, vy={vy:.2f}, yaw={yaw:.2f}")


# ============================================================
# 2. 物理动作原语（已修复步数截断 bug）
# ============================================================
def _run_steps(vx: float, vy: float, total_distance: float,
               speed: float, step_dist: float, direction_name: str):
    """
    通用小步执行器：
      - 步数向上取整（ceil），保证步数足以覆盖全程，不再向下截断丢距离；
      - 把 total_distance 均匀分摊到每一步（actual_step = total/count），
        保证 step_count × actual_step == total_distance 严格成立，
        每格位移无截断误差；
      - 减 1e-9 是为了消除浮点误差导致恰整除时多算一步
        （如 0.486/0.243 浮点结果可能是 2.0000000000000004）。
    """
    if total_distance <= 0:
        return
    step_count = max(1, math.ceil(total_distance / step_dist - 1e-9))
    actual_step = total_distance / step_count      # 每小步实际距离，可能略小于 step_dist
    single_sleep = actual_step / speed             # 每小步持续时间
    print(f"    {direction_name} {total_distance:.3f}m，"
          f"{step_count}×{actual_step:.3f}m小步")
    for _ in range(step_count):
        send_vel(vx, vy, 0.0)
        time.sleep(single_sleep)
    send_vel(0.0, 0.0, 0.0)
    time.sleep(0.3)


def move_forward(total_distance: float, speed=0.243, step_dist=0.243):
    """向前直行"""
    _run_steps(speed, 0.0, total_distance, speed, step_dist, "向前")


def move_backward(total_distance: float, speed=0.243, step_dist=0.243):
    """向后退"""
    _run_steps(-speed, 0.0, total_distance, speed, step_dist, "向后")


def move_left(total_distance: float, speed=0.243, step_dist=0.243):
    """向左横移（vy取正值）"""
    _run_steps(0.0, speed, total_distance, speed, step_dist, "向左")


def move_right(total_distance: float, speed=0.243, step_dist=0.243):
    """向右横移（vy取负值）"""
    _run_steps(0.0, -speed, total_distance, speed, step_dist, "向右")


def rotate_ccw(angle_deg: float, rot_speed=0.2):
    """逆时针原地旋转（保留，供以后调整朝向用）"""
    angle_rad = math.radians(angle_deg)
    rot_time = abs(angle_rad) / rot_speed
    print(f"    逆时针旋转 {angle_deg}°，持续{rot_time:.2f}秒")
    send_vel(0.0, 0.0, -rot_speed)
    time.sleep(rot_time)
    send_vel(0.0, 0.0, 0.0)
    time.sleep(0.3)


# ============================================================
# 3. 网格动作 → 真机动作映射
# ============================================================
CELL_SIZE = 0.6     # 一个网格 = 0.6 米

# 动作 → (真机执行函数, 网格坐标增量 dx/dy, 中文名)
ACTION_MAP = {
    0: ("上", move_forward,  (0, +1)),
    1: ("下", move_backward, (0, -1)),
    2: ("左", move_left,     (-1, 0)),
    3: ("右", move_right,    (+1, 0)),
}


def execute_grid_action(action: int):
    """
    执行一个网格动作：
      0=上 → 前进
      1=下 → 后退
      2=左 → 左横移
      3=右 → 右横移
    """
    if action not in ACTION_MAP:
        raise ValueError(f"非法动作: {action}")
    name, func, (dx, dy) = ACTION_MAP[action]
    print(f"    物理执行: {name} 一格（{CELL_SIZE}m）")
    func(CELL_SIZE)
    return dx, dy


# ============================================================
# 4. 地图 ASCII 可视化（可选，让用户直观看到机器狗在哪）
# ============================================================
def draw_map(env, x, y, goal, path=None):
    """
    在终端打印 ASCII 地图：
        # = 障碍物
        G = 终点
        S = 起点
        * = 历史路径
        @ = 当前位置
    """
    obstacles = set(env.obstacles)
    gx, gy = goal
    path_set = set(path) if path else set()

    print("\n    " + "".join(f"{i % 10}" for i in range(env.grid_size_x)))
    for row in range(env.grid_size_y - 1, -1, -1):
        line = f"{row:2d} |"
        for col in range(env.grid_size_x):
            if (col, row) == (x, y):
                line += "@"
            elif (col, row) == (gx, gy):
                line += "G"
            elif (col, row) in obstacles:
                line += "#"
            elif (col, row) in path_set:
                line += "*"
            else:
                line += "."
        print(line)


# ============================================================
# 5. 交互式读取起点
# ============================================================
def ask_start_position(env, goal):
    """让用户输入起点坐标，做范围/障碍物/终点校验"""
    gx, gy = goal
    max_x = env.grid_size_x - 1
    max_y = env.grid_size_y - 1
    obstacles = set(env.obstacles)

    print("\n" + "=" * 60)
    print("🧭 机器狗网格导航 - 请输入起点坐标")
    print("=" * 60)
    print(f"  网格范围: x ∈ [0, {max_x}], y ∈ [0, {max_y}]")
    print(f"  终点坐标: ({gx}, {gy})")
    print(f"  输入格式: '2 1' 或 '2,1'，输入 'q' 退出")
    print("=" * 60)

    while True:
        try:
            raw = input("\n起点坐标 >>> ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                return None, None

            raw = raw.replace(",", " ")
            parts = raw.split()
            if len(parts) != 2:
                print("❌ 格式错误，请输入两个数字，例如: 2 1")
                continue

            x, y = int(parts[0]), int(parts[1])

            if not (0 <= x <= max_x):
                print(f"❌ x 超出范围 [0, {max_x}]")
                continue
            if not (0 <= y <= max_y):
                print(f"❌ y 超出范围 [0, {max_y}]")
                continue
            if (x, y) in obstacles:
                print(f"❌ ({x}, {y}) 是障碍物")
                continue
            if (x, y) == (gx, gy):
                print(f"❌ ({x}, {y}) 是终点，不能作为起点")
                continue

            print(f"✅ 起点已确认: ({x}, {y})")
            return x, y

        except ValueError:
            print("❌ 请输入整数坐标")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 已取消")
            return None, None


# ============================================================
# 6. 主导航循环（核心）
# ============================================================
def navigate(start_x: int, start_y: int, max_steps: int = 200):
    """
    从 (start_x, start_y) 出发，循环调用模型决策 → 真机执行 → 更新坐标
    直到到达终点
    """
    # 触发模型加载，同时拿到 goal 和 env
    _ensure_loaded()
    import model_interface
    env = model_interface._ENV
    goal = model_interface._GOAL_POS
    gx, gy = goal

    print("\n" + "=" * 60)
    print(f"🚀 导航开始  起点=({start_x},{start_y})  终点=({gx},{gy})")
    print("=" * 60)

    x, y = start_x, start_y
    path = [(x, y)]

    # 展示初始位置
    draw_map(env, x, y, goal, path)

    for step in range(1, max_steps + 1):
        print(f"\n──── 第 {step:02d} 步 ────")
        print(f"  当前网格: ({x}, {y})")

        if (x, y) == (gx, gy):
            print("🎯 已到达终点！")
            break

        # ---- 1. 问模型 ----
        action = get_action_from_position(x, y)
        name = ACTION_MAP[action][0]
        print(f"  模型决策: {action} ({name})")

        # ---- 2. 真机执行 ----
        dx, dy = execute_grid_action(action)

        # ---- 3. 更新网格坐标 ----
        nx, ny = x + dx, y + dy

        # 安全兜底
        if (nx < 0 or nx >= env.grid_size_x or
                ny < 0 or ny >= env.grid_size_y or
                (nx, ny) in set(env.obstacles)):
            print(f"  ⚠️ 预测会撞墙/越界到 ({nx},{ny})，本次移动取消")
            send_vel(0.0, 0.0, 0.0)
            time.sleep(0.3)
        else:
            x, y = nx, ny

        path.append((x, y))
        print(f"  ✅ 执行完毕，当前网格: ({x}, {y})")

        # 每步画一次地图（地图大时可能会刷屏，如不想刷可以把这行注释掉）
        draw_map(env, x, y, goal, path)

    else:
        print(f"\n⚠️ 达到最大步数 {max_steps} 仍未到终点")

    # ---- 轨迹汇总 ----
    print("\n" + "=" * 60)
    print(f"📋 本次完整轨迹（共 {len(path)} 个格点）:")
    print("  " + " → ".join(f"({px},{py})" for px, py in path))
    print("=" * 60)

    return path


# ============================================================
# 7. 主入口
# ============================================================
if __name__ == "__main__":
    try:
        # 先加载模型拿到 env / goal
        _ensure_loaded()
        import model_interface
        env = model_interface._ENV
        goal = model_interface._GOAL_POS

        # 交互式输入起点
        sx, sy = ask_start_position(env, goal)

        if sx is not None:
            confirm = input(
                f"\n⚠️ 即将控制真机，从 ({sx},{sy}) 出发。"
                f"确认输 y，取消输 n: "
            ).strip().lower()

            if confirm == "y":
                navigate(sx, sy)
            else:
                print("🚫 已取消")

    except KeyboardInterrupt:
        print("\n⏹ 手动中断")
    finally:
        send_vel(0.0, 0.0, 0.0)
        time.sleep(0.3)
        udp_sock.close()
        print("🔌 UDP 已关闭，机器狗停机")