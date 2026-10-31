"""
eval_with_popup.py
功能：
  1) 弹窗让用户输入起点坐标 (x, y)；
  2) 弹窗让用户输入终点坐标 (x, y)（校验：不越界、不在障碍物、不等于起点）；
  3) 用 PPO 模型从起点跑到终点：录制视频 + 记录路径；
  4) 输出：
       [x,y].mp4                  —— 本局视频（以起点命名）
       [x,y]_decision_map.png     —— 决策地图（最优动作 + 路径）
       [x,y]_prob_maps.png        —— 四动作概率热力图
"""
import sys
import os
import re
import random
from pathlib import Path
from collections import deque

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pygame
import imageio
import torch
from stable_baselines3 import PPO
from env_module.GridEnv import GridNavEnv

# -------- tkinter（没有就降级为命令行输入） --------
try:
    import tkinter as tk
    from tkinter import simpledialog, messagebox
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

# ================== 中文字体 ==================
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC',
    'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
    'AR PL UMing CN', 'Droid Sans Fallback', 'Arial Unicode MS'
]
plt.rcParams['axes.unicode_minus'] = False

# ================== 路径 ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "ppo_grid_model.zip"
OUT_DIR      = Path(__file__).resolve().parent
OPEN_RESULTS = True


# ============================================================
# 通用工具
# ============================================================
def get_obstacles_set(env):
    return set(getattr(env, 'obstacles_set', env.obstacles))


def parse_coords(s):
    """宽松解析：支持 '3,5' '3 5' '(3,5)' '3，5' '【3,5】' 等格式。"""
    parts = [p for p in re.split(r'[,，\s()（）\[\]【】]+', s.strip()) if p]
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


# ============================================================
# 1. 弹窗输入起点
# ============================================================
def ask_start_point(env):
    obstacles = get_obstacles_set(env)
    x_max = env.grid_size_x - 1
    y_max = env.grid_size_y - 1

    if TK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        prompt = (f"请输入【起点】坐标 (x, y)\n\n"
                  f"x 范围：0 ~ {x_max}\n"
                  f"y 范围：0 ~ {y_max}\n\n"
                  f"示例：3,5   或   (3,5)")

        while True:
            s = simpledialog.askstring("设置起点", prompt, parent=root)
            if s is None:                      # 用户点了取消
                root.destroy()
                return None

            coord = parse_coords(s)
            if coord is None:
                messagebox.showerror("格式错误",
                                     "请输入两个整数，例如 3,5", parent=root)
                continue

            x, y = coord
            if not (0 <= x <= x_max and 0 <= y <= y_max):
                messagebox.showerror(
                    "坐标越界",
                    f"x 应在 0 ~ {x_max}，y 应在 0 ~ {y_max}", parent=root)
                continue

            if (x, y) in obstacles:
                messagebox.showerror(
                    "障碍物",
                    f"({x},{y}) 是障碍物，请换一个坐标", parent=root)
                continue

            root.destroy()
            return (x, y)

    # ---- 命令行 fallback ----
    print(f"[无 tkinter，使用命令行输入] x∈[0,{x_max}], y∈[0,{y_max}]")
    while True:
        s = input("起点 (x,y): ")
        coord = parse_coords(s)
        if coord is None:
            print("格式错误，请重试")
            continue
        x, y = coord
        if not (0 <= x <= x_max and 0 <= y <= y_max):
            print("超出地图范围")
            continue
        if (x, y) in obstacles:
            print("该坐标是障碍物")
            continue
        return (x, y)


# ============================================================
# 2. 弹窗输入终点（额外校验：不能等于起点）
# ============================================================
def ask_goal_point(env, start):
    obstacles = get_obstacles_set(env)
    x_max = env.grid_size_x - 1
    y_max = env.grid_size_y - 1

    if TK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        prompt = (f"请输入【终点】坐标 (x, y)\n\n"
                  f"起点是：{start}  （终点不能与起点相同）\n"
                  f"x 范围：0 ~ {x_max}\n"
                  f"y 范围：0 ~ {y_max}\n\n"
                  f"示例：3,5   或   (3,5)")

        while True:
            s = simpledialog.askstring("设置终点", prompt, parent=root)
            if s is None:
                root.destroy()
                return None

            coord = parse_coords(s)
            if coord is None:
                messagebox.showerror("格式错误",
                                     "请输入两个整数，例如 3,5", parent=root)
                continue

            x, y = coord
            if not (0 <= x <= x_max and 0 <= y <= y_max):
                messagebox.showerror(
                    "坐标越界",
                    f"x 应在 0 ~ {x_max}，y 应在 0 ~ {y_max}", parent=root)
                continue

            if (x, y) in obstacles:
                messagebox.showerror(
                    "障碍物",
                    f"({x},{y}) 是障碍物，请换一个坐标", parent=root)
                continue

            if (x, y) == tuple(start):
                messagebox.showerror(
                    "与起点重合",
                    f"终点不能和起点 {tuple(start)} 相同，请换一个坐标", parent=root)
                continue

            root.destroy()
            return (x, y)

    # ---- 命令行 fallback ----
    print(f"[无 tkinter，使用命令行输入] 起点={tuple(start)}，"
          f"x∈[0,{x_max}], y∈[0,{y_max}]")
    while True:
        s = input("终点 (x,y): ")
        coord = parse_coords(s)
        if coord is None:
            print("格式错误，请重试")
            continue
        x, y = coord
        if not (0 <= x <= x_max and 0 <= y <= y_max):
            print("超出地图范围")
            continue
        if (x, y) in obstacles:
            print("该坐标是障碍物")
            continue
        if (x, y) == tuple(start):
            print("终点不能和起点相同")
            continue
        return (x, y)


# ============================================================
# 3. 录制一局（从已设置好的 agent_pos / goal_pos 出发）
# ============================================================
def record_episode_from_state(env, model, obs, fps=10, max_frames=2000):
    start = tuple(env.agent_pos)
    goal  = tuple(env.goal_pos)
    path  = [start]
    frames = []
    reached = False

    def capture():
        return pygame.surfarray.array3d(env.window).transpose([1, 0, 2])

    while len(frames) < max_frames:
        env.render()
        frames.append(capture())

        # 事件处理（关窗口）
        quit_flag = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_flag = True
                break
        if quit_flag:
            break

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        path.append(tuple(env.agent_pos))

        if terminated or truncated:
            reached = bool(terminated)
            # 补抓最后一帧（真正站在终点的画面）
            if len(frames) < max_frames:
                env.render()
                frames.append(capture())
            break

    return {
        "frames": frames, "path": path,
        "start": start, "goal": goal,
        "reached": reached, "fps": fps,
    }


# ============================================================
# 4. 策略可视化（基于本局固定的 start / goal）
# ============================================================
def visualize_policy(env, model, start, goal, path, out_prefix):
    grid_x, grid_y = env.grid_size_x, env.grid_size_y
    obstacles = get_obstacles_set(env)

    # ---- 动作概率 ----
    action_probs = np.zeros((grid_y, grid_x, 4))
    for y in range(grid_y):
        for x in range(grid_x):
            if (x, y) in obstacles:
                continue
            up    = 0 if (y + 1 >= grid_y or (x, y + 1) in obstacles) else 1
            down  = 0 if (y - 1 < 0       or (x, y - 1) in obstacles) else 1
            left  = 0 if (x - 1 < 0       or (x - 1, y) in obstacles) else 1
            right = 0 if (x + 1 >= grid_x or (x + 1, y) in obstacles) else 1

            obs = np.array([x, y, goal[0], goal[1], up, down, left, right],
                           dtype=np.float32)
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
            action_probs[y, x] = probs

    best_actions = np.argmax(action_probs, axis=-1)

    action_names = ['↑ 上', '↓ 下', '← 左', '→ 右']
    cmap = ListedColormap(['#377eb8', '#e41a1c', '#4daf4a', '#984ea3'])

    # ---------------- 图 1：决策地图 ----------------
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(best_actions, origin='lower', cmap=cmap, vmin=-0.5, vmax=3.5)

    for (ox, oy) in obstacles:
        ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1,
                                   facecolor='black', edgecolor='black'))

    ax.plot(start[0], start[1], 'wo', markersize=10, markeredgecolor='black')
    ax.plot(goal[0],  goal[1],  'y*', markersize=15, markeredgecolor='black')

    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    if len(path) > 1:
        ax.plot(path_x, path_y, 'r--', linewidth=2.5, label='行走路径')

    dirs = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    arrow_len = 0.3
    for y in range(grid_y):
        for x in range(grid_x):
            if (x, y) in obstacles:
                continue
            dx, dy = dirs[best_actions[y, x]]
            ax.arrow(x, y, dx * arrow_len, dy * arrow_len,
                     head_width=0.15, head_length=0.15,
                     fc='white', ec='black', linewidth=1,
                     alpha=0.8, length_includes_head=True)

    xticks = np.arange(0, grid_x, 5)
    yticks = np.arange(0, grid_y, 5)
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    ax.set_xticklabels(xticks); ax.set_yticklabels(yticks)
    ax.set_xlabel('X 坐标'); ax.set_ylabel('Y 坐标')
    ax.set_title(f'PPO 决策地图 (起点 {start}, 终点 {goal})', fontsize=15)

    patches = [mpatches.Patch(color=cmap(i), label=action_names[i]) for i in range(4)]
    patches.append(mpatches.Patch(color='none', label='___ 路径 (红色虚线)'))
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_xticks(np.arange(-0.5, grid_x, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_y, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)

    fig.tight_layout()
    decision_path = f"{out_prefix}_decision_map.png"
    fig.savefig(decision_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---------------- 图 2：动作概率热力图 ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(action_probs[:, :, i], origin='lower',
                       cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'动作 "{action_names[i]}" 的概率')
        ax.set_xticks([]); ax.set_yticks([])
        for (ox, oy) in obstacles:
            ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1,
                                       facecolor='black', edgecolor='black'))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'各动作概率分布 (起点 {start}, 终点 {goal})', fontsize=15)
    fig.tight_layout()
    prob_path = f"{out_prefix}_prob_maps.png"
    fig.savefig(prob_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return decision_path, prob_path


# ============================================================
# 主流程
# ============================================================
def main():
    env = None
    try:
        if not MODEL_PATH.exists():
            print(f"❌ 模型文件不存在：{MODEL_PATH}")
            print("   请先运行 ppo_train.py 生成 models/ppo_grid_model.zip")
            return

        env = GridNavEnv()
        model = PPO.load(str(MODEL_PATH))
        print(f"✅ 模型加载成功：{MODEL_PATH}")

        # ---- 1. 弹窗输入起点 ----
        start = ask_start_point(env)
        if start is None:
            print("已取消输入起点，退出。")
            env.close()
            return

        # ---- 2. 弹窗输入终点（校验不等于起点） ----
        goal = ask_goal_point(env, start)
        if goal is None:
            print("已取消输入终点，退出。")
            env.close()
            return

        # ---- 3. 设置起点 + 终点 ----
        env.agent_pos = list(start)
        env.goal_pos  = list(goal)
        obs = env._get_obs()

        print(f"起点：{start}")
        print(f"终点：{goal}")

        # ---- 4. 录制一局 ----
        result = record_episode_from_state(env, model, obs)
        print(f"录制帧数：{len(result['frames'])}，"
              f"路径长度：{len(result['path']) - 1} 步，"
              f"到达终点：{result['reached']}")

        # ---- 5. 生成策略映射图 ----
        out_prefix = str(OUT_DIR / f"[{start[0]},{start[1]}]")
        print("正在计算整张地图策略并生成映射图 ...")
        dec_path, prob_path = visualize_policy(
            env, model, start, goal, result["path"], out_prefix
        )
        print(f"✅ 决策地图：{dec_path}")
        print(f"✅ 概率图  ：{prob_path}")

        env.close()
        env = None

        # ---- 6. 导出视频 ----
        mp4_path = f"{out_prefix}.mp4"
        if result["frames"]:
            print(f"正在合成视频：{mp4_path}，帧数 {len(result['frames'])}，"
                  f"帧率 {result['fps']}")
            imageio.mimsave(mp4_path, result["frames"], fps=result["fps"])
            print(f"✅ 视频：{mp4_path}")
        else:
            print("⚠️ 没有录到任何帧，跳过视频生成。")

        # ---- 7. 自动打开结果 ----
        if OPEN_RESULTS:
            for p in (mp4_path, dec_path, prob_path):
                if os.path.exists(p):
                    try:
                        os.startfile(p)
                    except Exception as e:
                        print(f"（自动打开失败：{p}）原因：{e}")

    except pygame.error as e:
        print(f"Pygame 异常：{e}")
        print("解决：关闭所有终端，重新打开再运行脚本")
        if env is not None:
            env.close()
    except KeyboardInterrupt:
        print("\n手动中断录制，正在关闭窗口...")
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()