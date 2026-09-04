# S3.1 补充图：B2 学习核（稀疏团块/中心-外周）vs B5 Gabor（方向条纹）聚焦对比
# 输出：results/fig2b_morphology_compare.png
import sys, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'IDEA1-covstdp/src')
from gabor_frontend import gabor_kernel_bank

B2_MODEL = 'vprtempo/models/b2_500_block2__seed0.pth'
sd = torch.load(B2_MODEL, map_location='cpu', weights_only=True)['model_0']
w_b2 = sd['conv_layer.w.weight'].detach().clone()          # [32,1,5,5]
exc  = sd['conv_layer.havconnExc'].clone()
w_g  = gabor_kernel_bank()                                  # [32,1,5,5]

on_idx  = [i for i in range(32) if exc[i]][:4]
off_idx = [i for i in range(32) if not exc[i]][:2]
b2_sel  = on_idx + off_idx
b5_sel  = [0, 8, 16, 24, 4, 12]

def norm(k):
    k = k - k.mean()
    m = k.abs().max()
    return k / m if m > 0 else k

fig, axes = plt.subplots(2, 6, figsize=(14, 6))
for j, i in enumerate(b2_sel):
    ax = axes[0, j]
    ax.imshow(norm(w_b2[i, 0]), cmap='RdBu_r', vmin=-1, vmax=1)
    tag = 'ON' if exc[i] else 'OFF'
    ax.set_title(f'B2 ch{i} ({tag})', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
for j, i in enumerate(b5_sel):
    ax = axes[1, j]
    ax.imshow(norm(w_g[i, 0]), cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'B5 ch{i}', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

fig.text(0.02, 0.68, 'B2 (Conv-STDP learned)\nsparse blob /\ncenter-surround',
         fontsize=11, fontweight='bold', va='center')
fig.text(0.02, 0.28, 'B5 (hand-crafted)\nGabor bank\noriented stripes',
         fontsize=11, fontweight='bold', va='center')
fig.suptitle('What single-step competitive Hebbian learns vs what the task rewards '
             '(red = positive, blue = negative)', fontsize=13)
fig.text(0.5, 0.02, 'Note: sign-constrained ON kernels (all-positive weights) cannot express the +/- lobe alternation '
                    'that oriented stripes require; sparse blobs / ridges are the predicted morphology.',
         ha='center', fontsize=9, style='italic')
plt.tight_layout(rect=[0.09, 0.06, 1, 0.92])
plt.savefig('IDEA1-covstdp/results/fig2b_morphology_compare.png', dpi=180)
print('saved')
