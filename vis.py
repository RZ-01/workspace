import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np


def crop_xy(img, max_xy=1300):
    """裁剪图像的 y/x 轴到 [0, max_xy)"""
    if img.ndim < 2:
        return img
    h = min(max_xy, img.shape[0])
    w = min(max_xy, img.shape[1])
    return img[:h, :w]

def normalize_image(img):
    """将图像归一化到 [0, 1]"""
    img_min = np.min(img)
    img_max = np.max(img)
    # 防止除以 0
    if img_max - img_min == 0:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)

# 1. 文件路径配置
file_paths = {
    "NGP_Clear": "/workspace/inference/lsm_mouse_heart_constantLR_clear_z01050.tif",
    "NGLOD_Clear": "/workspace/inference/nglod_mouse_heart_clear_z01050.tif",
    "NGP_Blurred": "/workspace/inference/lsm_mouse_heart_constantLR_blurred_z01050.tif",
    "NGLOD_Blurred": "/workspace/inference/nglod_mouse_heart_blurred_z01050.tif"
}

# 2. 读取并归一化图像
images = {}
for name, path in file_paths.items():
    try:
        data = tiff.imread(path).astype(np.float32)
        data = crop_xy(data, max_xy=1300)
        images[name] = normalize_image(data)
    except Exception as e:
        print(f"无法读取文件 {path}: {e}")
        images[name] = np.zeros((1300, 1300), dtype=np.float32) # 占位图

# 3. 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
plt.subplots_adjust(wspace=0.1, hspace=0.25)

# 标题设置
fig.suptitle('Comparison of NGP and NGLOD (Normalized to [0, 1])\nSlice z=300', 
             fontsize=16, fontweight='bold', y=0.95)

# --- 第一行：Clear 对比 ---
axes[0, 0].imshow(images["NGP_Clear"], cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title("NGP: Clear Image", fontsize=12, pad=10)
axes[0, 0].set_ylabel("CLEAR", fontsize=14, fontweight='bold', labelpad=20)

axes[0, 1].imshow(images["NGLOD_Clear"], cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title("NGLOD: Clear Image", fontsize=12, pad=10)

# --- 第二行：Blurred 对比 ---
axes[1, 0].imshow(images["NGP_Blurred"], cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title("NGP: Blurred Image", fontsize=12, pad=10)
axes[1, 0].set_ylabel("BLURRED", fontsize=14, fontweight='bold', labelpad=20)

axes[1, 1].imshow(images["NGLOD_Blurred"], cmap='gray', vmin=0, vmax=1)
axes[1, 1].set_title("NGLOD: Blurred Image", fontsize=12, pad=10)

# 移除所有坐标轴刻度
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# 添加 Colorbar 显示归一化范围
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap='gray', norm=plt.Normalize(vmin=0, vmax=1))
fig.colorbar(sm, cax=cbar_ax, label='Normalized Intensity')

plt.savefig("/workspace/comparison_figure.pdf", dpi=300, bbox_inches='tight')