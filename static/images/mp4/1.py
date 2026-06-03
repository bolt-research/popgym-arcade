import os
from moviepy import VideoFileClip, clips_array
from PIL import Image
import numpy as np
import random

gif_folder = "/home/admuser/icyzhe/code/webpage/popgym-arcade/static/images/mp4"
output_file = "output.mp4"
cols, rows = 10, 5         
target_duration = 30   
target_fps = 24



gif_files = sorted([
    f for f in os.listdir(gif_folder)
    if f.lower().endswith('.gif')
])
print(f"找到 {len(gif_files)} 个 GIF，将循环使用填满 {cols}x{rows} 网格")

base_clips = []
for f in gif_files:
    path = os.path.join(gif_folder, f)
    clip = VideoFileClip(path, has_mask=False)
    if clip.duration < target_duration:
        clip = clip.loop(duration=target_duration)
    else:
        clip = clip.subclipped(0, target_duration)
    clip = clip.with_fps(target_fps)
    base_clips.append(clip)

# 如果所有 GIF 尺寸不统一，这里强制统一（取第一个的尺寸）
width, height = base_clips[0].size
base_clips = [c.resized(new_size=(width, height)) for c in base_clips]
random.shuffle(base_clips)
# 构造 5x10 网格，循环取用 base_clips 中的 GIF
grid_clips = []
clip_index = 0
for r in range(rows):
    row_clips = []
    for c in range(cols):
        # 循环使用已有的 GIF
        row_clips.append(base_clips[clip_index % len(base_clips)])
        clip_index += 1
    grid_clips.append(row_clips)

# 合成网格视频
final_clip = clips_array(grid_clips)

# 导出 MP4
final_clip.write_videofile(
    output_file,
    codec='libx264',
    fps=target_fps,
    audio=False,
    preset='medium',
    threads=4
)

print(f"✅ 视频已生成：{output_file}")