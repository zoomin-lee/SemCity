import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def seed_all(seed):
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def draw_scalar_field2D(arr, vmin=None, vmax=None, cmap=None, title=None):
    multi = max(arr.shape[0] // 512, 1)
    fig, ax = plt.subplots(figsize=(5 * multi, 5 * multi))
    cax1 = ax.matshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(cax1, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if title is not None:
        ax.set_title('08/'+str(title).zfill(6))
    return fig

def get_result(evaluator, class_name):
    _, class_jaccard = evaluator.getIoU()
    m_jaccard = class_jaccard[1:].mean()
    miou = m_jaccard * 100
    conf = evaluator.get_confusion()
    iou = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0] + 1e-8) * 100
    evaluator.reset()
    
    print(f"mIoU: {miou:.2f}")
    print(f"iou: {iou:.2f}")

    for i, c in enumerate(class_name) :
        print(f"{c}: {class_jaccard[i]*100:.2f}")