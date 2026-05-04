import cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

gt   = cv2.imread('/workspace/temp/W_DIP/datasets/levin/gt/im1.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
blur = cv2.imread('/workspace/temp/W_DIP/datasets/levin/blur/im1_kernel2_img.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
wdip = cv2.imread('/workspace/temp/W_DIP/results/levin/WDIP/im1_kernel2_img_x.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
inferred = cv2.imread('/workspace/temp/workspace/inference_2d/inferred.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.

comparisons = [('blurred input', blur), ('WDIP', wdip), ('our inferred', inferred)]
scores = []
for name, img in comparisons:
    p = psnr(gt, img, data_range=1.)
    s = ssim(gt, img, data_range=1.)
    scores.append((name, p, s))
    print(f'{name:15s}  PSNR={p:.2f}  SSIM={s:.4f}')

fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

axes[0, 0].imshow(gt, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('Ground truth')
axes[1, 0].axis('off')

for idx, ((name, img), (_, p, s)) in enumerate(zip(comparisons, scores), start=1):
    axes[0, idx].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, idx].set_title(f'{name}\nPSNR={p:.2f}, SSIM={s:.4f}')

    diff = np.abs(gt - img)
    im = axes[1, idx].imshow(diff, cmap='magma', vmin=0, vmax=1)
    axes[1, idx].set_title(f'|GT - {name}|')

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes[1, 1:], fraction=0.04, pad=0.02)
cbar.set_label('Absolute error')

output_dir = Path('/workspace/temp/workspace/inference_2d')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'comparison_plot.png'
fig.savefig(output_path, dpi=200)
print(f'Saved visualization: {output_path}')