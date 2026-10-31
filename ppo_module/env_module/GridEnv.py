"""
目标：从随机起点到随机终点，用表格 Q-learning 学会导航
坐标系：原点在左下角，x向右增加，y向上增加

结构：
  1. GridNavEnv        —— 环境（去掉外圈、起点终点都随机、互不重合、终点可达）
  2. train_q_learning  —— 表格 Q-learning 训练
  3. evaluate          —— 贪心策略评估成功率
  4. demo              —— 贪心策略可视化跑一局
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from collections import deque, defaultdict


# ==============================================================
# 1. 环境
# ==============================================================
class GridNavEnv(gym.Env):
    # 训练时不会调用 render；演示时 30fps 比较流畅（可自行改）
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # ---- 原始地图尺寸（16 列 × 20 行，坐标 x:0~15, y:0~19） ----
        raw_size_x, raw_size_y = 16, 20

        # ---- 去掉外围一圈 ----
        self.grid_size_x = raw_size_x - 2   # 14 -> x: 0~13
        self.grid_size_y = raw_size_y - 2   # 18 -> y: 0~17

        # ---- 渲染 ----
        self.cell_size = 40
        self.width = self.grid_size_x * self.cell_size   # 560
        self.height = self.grid_size_y * self.cell_size  # 720

        # ---- 动作 / 观测空间 ----
        self.action_space = spaces.Discrete(4)   # 0上 1下 2左 3右
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.grid_size_x, self.grid_size_y) - 1,
            shape=(8,),
            dtype=np.float32
        )

        # ---- 原始障碍物（原始坐标系） ----
        raw_obstacles = [
            (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(0,10),(0,11),(0,12),(0,13),(0,14),
            (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(1,10),(1,11),(1,12),(1,13),(1,14),
            (2,0),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(2,10),(2,11),(2,12),(2,13),(2,14),
            (3,0),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(3,18),(3,19),
            (4,0),(4,4),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(4,11),(4,18),(4,19),
            (5,0),(5,4),(5,5),(5,6),(5,7),(5,8),(5,9),(5,10),(5,11),(5,15),(5,16),(5,17),(5,18),(5,19),
            (6,0),(6,4),(6,5),(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),(6,15),(6,16),(6,17),(6,18),(6,19),
            (7,0),(7,1),(7,15),(7,16),(7,17),(7,18),(7,19),
            (8,0),(8,1),(8,15),(8,16),(8,17),(8,18),(8,19),
            (9,0),(9,1),(9,15),(9,16),(9,17),(9,18),(9,19),
            (10,0),(10,1),(10,2),(10,15),(10,16),(10,17),(10,18),(10,19),
            (11,0),(11,1),(11,2),(11,15),(11,16),(11,17),(11,18),(11,19),
            (12,0),(12,1),(12,2),(12,18),(12,19),
            (13,12),
            (15,4),(15,5),(15,6),(15,7),(15,10),(15,18),(15,19)
        ]

        # ---- 自动裁剪：保留内圈 + 平移 (-1, -1) ----
        self.obstacles = [
            (x - 1, y - 1) for (x, y) in raw_obstacles
            if 1 <= x <= raw_size_x - 2 and 1 <= y <= raw_size_y - 2
        ]
        self.obstacles_set = set(self.obstacles)

        # ---- 所有空白格（合法起点/终点候选池） ----
        self.free_cells = [
            (x, y) for x in range(self.grid_size_x)
            for y in range(self.grid_size_y)
            if (x, y) not in self.obstacles_set
        ]

        # ---- 预计算每个起点的可达集 / 目标候选池（加速 reset） ----
        self._reachable_cache = {}
        self._goal_cache = {}
        for cell in self.free_cells:
            reachable = self._bfs_reachable(cell)
            self._reachable_cache[cell] = reachable
            cand = [p for p in reachable if p != cell]
            if not cand:   # 兜底：起点被完全孤立（本图不会触发）
                cand = [p for p in self.free_cells if p != cell]
            self._goal_cache[cell] = cand

        # 占位，reset 时重新赋值
        self.agent_pos = [0, 0]
        self.goal_pos = [1, 0]

        self.window = None
        self.clock = None

    # ----------------------------------------------------------
    # BFS：从 start 出发可达的所有格子（含 start 自身）
    # ----------------------------------------------------------
    def _bfs_reachable(self, start):
        start = tuple(start)
        visited = {start}
        q = deque([start])
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_size_x and
                        0 <= ny < self.grid_size_y and
                        (nx, ny) not in self.obstacles_set and
                        (nx, ny) not in visited):
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return visited

    # ----------------------------------------------------------
    # reset：起点、终点都随机（空白格、不重合、终点可达）
    # ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = list(random.choice(self.free_cells))
        start_tuple = tuple(self.agent_pos)

        candidates = self._goal_cache[start_tuple]   # 可达 且 ≠ 起点
        self.goal_pos = list(random.choice(candidates))

        return self._get_obs(), {}

    # ----------------------------------------------------------
    # 观测（8 维）：x, y, gx, gy, 上/下/左/右 可走性
    # ----------------------------------------------------------
    def _get_obs(self):
        x, y = self.agent_pos
        gx, gy = self.goal_pos

        up    = 0 if (y + 1 >= self.grid_size_y or (x, y + 1) in self.obstacles_set) else 1
        down  = 0 if (y - 1 < 0 or (x, y - 1) in self.obstacles_set) else 1
        left  = 0 if (x - 1 < 0 or (x - 1, y) in self.obstacles_set) else 1
        right = 0 if (x + 1 >= self.grid_size_x or (x + 1, y) in self.obstacles_set) else 1

        return np.array([x, y, gx, gy, up, down, left, right], dtype=np.float32)

    # ----------------------------------------------------------
    # 曼哈顿距离
    # ----------------------------------------------------------
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # ----------------------------------------------------------
    # step
    # ----------------------------------------------------------
    def step(self, action):
        old_dist = self._manhattan_distance(self.agent_pos, self.goal_pos)

        new_pos = list(self.agent_pos)
        if action == 0:    new_pos[1] += 1   # 上
        elif action == 1:  new_pos[1] -= 1   # 下
        elif action == 2:  new_pos[0] -= 1   # 左
        elif action == 3:  new_pos[0] += 1   # 右

        terminated = False
        reward = 0.0

        hit = (new_pos[0] < 0 or new_pos[0] >= self.grid_size_x or
               new_pos[1] < 0 or new_pos[1] >= self.grid_size_y or
               tuple(new_pos) in self.obstacles_set)

        if hit:
            reward = -5.0
        else:
            self.agent_pos = new_pos
            new_dist = self._manhattan_distance(self.agent_pos, self.goal_pos)
            reward = (old_dist - new_dist) * 2.0
            reward -= 0.05

        if self.agent_pos == self.goal_pos:
            reward = 200.0
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    # ----------------------------------------------------------
    # render
    # ----------------------------------------------------------
    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Grid Navigation · Q-learning")
            self.clock = pygame.time.Clock()

        pygame.event.pump()
        self.window.fill((255, 255, 255))

        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.window, (220, 220, 220), (x, 0), (x, self.height))
        for y_logical in range(self.grid_size_y):
            screen_y = (self.grid_size_y - 1 - y_logical) * self.cell_size
            pygame.draw.line(self.window, (220, 220, 220), (0, screen_y), (self.width, screen_y))

        for obs in self.obstacles:
            x, y = obs
            screen_x = x * self.cell_size
            screen_y = (self.grid_size_y - 1 - y) * self.cell_size
            pygame.draw.rect(self.window, (0, 0, 0),
                             pygame.Rect(screen_x, screen_y, self.cell_size, self.cell_size))

        gx, gy = self.goal_pos
        pygame.draw.rect(self.window, (231, 76, 60),
                         pygame.Rect(gx * self.cell_size,
                                     (self.grid_size_y - 1 - gy) * self.cell_size,
                                     self.cell_size, self.cell_size))

        ax, ay = self.agent_pos
        pygame.draw.rect(self.window, (46, 204, 113),
                         pygame.Rect(ax * self.cell_size,
                                     (self.grid_size_y - 1 - ay) * self.cell_size,
                                     self.cell_size, self.cell_size))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()


# ==============================================================
# 2. 表格 Q-learning
# ==============================================================
def obs_to_state(obs):
    """状态 = (x, y, gx, gy)，四方向可走性由 (x,y) 唯一决定，不必入状态。"""
    return (int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3]))


def train_q_learning(env,
                     episodes=20000,
                     max_steps=300,
                     alpha=0.1,
                     gamma=0.99,
                     eps_start=1.0,
                     eps_end=0.05,
                     eps_decay_episodes=10000,
                     eval_every=2000,
                     eval_episodes=100):
    """
    返回 Q 表（defaultdict: state -> np.array(4)）。
    说明：
      - 撞墙或到达终点时 target = r（不 bootstrap），避免 Q 值被 -5 无限拉低；
      - 其余情况 target = r + gamma * max_a' Q(s', a')。
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    recent_success = deque(maxlen=1000)
    recent_reward = deque(maxlen=1000)

    for ep in range(1, episodes + 1):
        # 线性衰减 epsilon
        eps = max(eps_end,
                  eps_start - (eps_start - eps_end) * (ep - 1) / eps_decay_episodes)

        obs, _ = env.reset()
        total_r = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            s = obs_to_state(obs)
            # epsilon-greedy
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(Q[s]))

            obs2, r, term, trunc, _ = env.step(a)
            s2 = obs_to_state(obs2)

            # 撞墙时 s2 == s 且未终止；到达终点时 term=True
            if term or s2 == s:
                target = r
            else:
                target = r + gamma * np.max(Q[s2])

            Q[s][a] += alpha * (target - Q[s][a])

            obs = obs2
            total_r += r
            steps += 1
            done = term

        recent_success.append(1.0 if done else 0.0)
        recent_reward.append(total_r)

        if ep % eval_every == 0:
            eval_success = evaluate(env, Q, n_episodes=eval_episodes, max_steps=max_steps)
            print(f"Ep {ep:6d} | eps={eps:.3f} | "
                  f"训练窗口成功率={np.mean(recent_success):.2%} "
                  f"平均回报={np.mean(recent_reward):7.1f} | "
                  f"贪心评估成功率={eval_success:.2%} | "
                  f"Q表大小={len(Q)}")

    return Q


# ==============================================================
# 3. 贪心策略评估
# ==============================================================
def evaluate(env, Q, n_episodes=100, max_steps=300):
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            s = obs_to_state(obs)
            a = int(np.argmax(Q[s]))
            obs, r, term, trunc, _ = env.step(a)
            done = term
            steps += 1
        if done:
            successes += 1
    return successes / n_episodes


# ==============================================================
# 4. 可视化演示
# ==============================================================
def demo(env, Q, max_steps=300):
    obs, _ = env.reset()
    print(f"demo 起始：start={env.agent_pos}, goal={env.goal_pos}")
    done = False
    steps = 0
    while not done and steps < max_steps:
        env.render()
        s = obs_to_state(obs)
        a = int(np.argmax(Q[s]))
        obs, r, term, trunc, _ = env.step(a)
        done = term
        steps += 1
    print(f"demo 结束：steps={steps}, agent={env.agent_pos}, "
          f"goal={env.goal_pos}, 到达={done}")
    env.close()


# ==============================================================
# 主程序
# ==============================================================
if __name__ == "__main__":
    # 想可复现就打开这两行（需放在创建 env 之前）
    # random.seed(42)
    # np.random.seed(42)

    env = GridNavEnv()
    print(f"地图尺寸：{env.grid_size_x} × {env.grid_size_y}")
    print(f"空白格子数：{len(env.free_cells)}")
    print(f"障碍物数：{len(env.obstacles)}")
    print("-" * 70)

    # ---- 训练 ----
    Q = train_q_learning(env, episodes=20000)

    # ---- 最终评估 ----
    print("\n=== 训练完成，最终贪心评估 ===")
    final_success = evaluate(env, Q, n_episodes=200, max_steps=300)
    print(f"贪心策略 200 局成功率：{final_success:.2%}")

    # ---- 可视化一局 ----
    print("\n=== 可视化一局贪心策略（关闭窗口或跑完自动退出）===")
    demo(env, Q)