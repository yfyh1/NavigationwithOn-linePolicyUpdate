import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pygame
import imageio
import os
from stable_baselines3 import PPO
from env_module.GridEnv import GridNavEnv

def main():
    env = None
    try:
        env = GridNavEnv()
        model = PPO.load("./models/ppo_grid_model.zip")
        obs, info = env.reset()
        
        # 获取当前随机起点坐标，构造文件名
        start_x, start_y = env.agent_pos
        mp4_name = f"[{start_x},{start_y}].mp4"
        
        frames = []
        max_episode_frames = 2000
        fps = 10
        print(f"当前起点坐标：({start_x},{start_y})，开始录制单轮Episode，到达终点自动停止")
        running = True
        
        while running and len(frames) < max_episode_frames:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            if not running:
                break
            
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            surf = env.window
            frame = pygame.surfarray.array3d(surf)
            frame = frame.transpose([1, 0, 2])
            frames.append(frame)
            
            # 到达终点/截断直接停止录制
            if terminated or truncated:
                print(f"单轮导航结束，总录制帧数：{len(frames)}")
                break
        
        if env is not None:
            env.close()
        
        # 导出以起点坐标命名的视频
        print(f"正在合成视频：{mp4_name}，总帧数 {len(frames)}，帧率 {fps}")
        imageio.mimsave(mp4_name, frames, fps=fps)
        print(f"✅ 视频生成完成：{mp4_name}")
        os.startfile(mp4_name)
        
    except pygame.error as e:
        print(f"Pygame视频系统异常：{e}")
        print("解决：关闭所有终端，重新打开再运行脚本")
        if env is not None:
            env.close()
    except KeyboardInterrupt:
        print("\n手动中断录制，正在关闭窗口...")
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()
