import sys
import os
# import yaml  

# 将项目根目录（env_module 的父目录）加入 Python 搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO          # PPO 算法实现（包含 PPO-Clip）
from env_module.GridEnv import GridNavEnv            # 自定义 21×16 栅格环境


# ============================================================
# 2. 主训练函数
# ============================================================
def main():
    # # ---- 读取超参数配置文件 ----
    # # 获取当前脚本所在目录
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # config_path = os.path.join(script_dir, "ppo_config.yaml")
    # with open(config_path, "r") as f:
    #     config = yaml.safe_load(f)
    # print("📋 已加载配置:", config)

    # ---- 2.1 创建模型保存目录 ----
    # os.makedirs 递归创建目录，exist_ok=True 表示目录已存在时不报错
    os.makedirs("models", exist_ok=True)

    # ---- 2.2 实例化环境 ----
    # 此时会调用 GridNavEnv.__init__()，加载地图尺寸、障碍物等
    env = GridNavEnv()

    # ---- 输出提示信息 ----
    print("正在初始化基于网格的强化学习agent（PPO）…")
    print("训练开始...")

    # ---- 2.3 创建 PPO 模型 ----
    # 参数说明：
    #   "MlpPolicy"      : 使用多层感知机（全连接网络）作为策略网络，输入为 4 维观测值（智能体坐标 + 目标坐标）
    #   env              : 训练环境实例
    #   verbose=1        : 日志详细程度，1 表示输出训练进度和关键指标（如 episode reward）
    #   learning_rate    : Adam 优化器的学习率（控制参数更新步长）
    #
    # 其他参数（如 n_steps, batch_size, ent_coef 等）均使用 Stable-Baselines3 的默认值：
    #   - n_steps=2048      : 每轮收集 2048 步经验后进行一次更新
    #   - batch_size=64     : 每次更新时的批量大小
    #   - n_epochs=10       : 每轮数据重复训练 10 次
    #   - gamma=0.99        : 折扣因子
    #   - gae_lambda=0.95   : GAE 平滑参数
    #   - clip_range=0.2    : PPO-Clip 的裁剪范围
    #   - ent_coef=0.0      : 熵系数（默认不鼓励探索）
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0005,
        ent_coef=0.2,
        tensorboard_log="./tensorboard_logs/"      #观察曲线：tensorboard --logdir=./tensorboard_logs/
    )

    # ---- 2.4 执行训练 ----
    # total_timesteps : 总交互步数（注意不是梯度更新次数，而是环境 step 的总次数）
    # 训练过程中会自动打印每轮迭代的统计信息（如平均奖励、episode 长度等）
    model.learn(total_timesteps=100000)            #修改为 数值 以获得更好的训练效果

    # ---- 2.5 保存模型 ----
    # 保存为 .zip 文件，包含网络权重和优化器状态，便于后续加载和继续训练
    model.save("models/ppo_grid_model")
    print("\n✅ 训练完成 & 模型已保存!")

    # ---- 2.6 关闭环境（释放 Pygame 等资源） ----
    env.close()

# ============================================================
# 3. 脚本入口：确保只在直接运行时执行训练，被导入时不执行
# ============================================================
if __name__ == "__main__":
    main()