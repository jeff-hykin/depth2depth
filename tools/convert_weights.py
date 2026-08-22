"""Convert official Depth Anything V2 metric vits weights to the two
safetensors files depth2depth loads.

Get the checkpoint (Hypersim indoor, max_depth=20) from
https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
then:

    python tools/convert_weights.py depth_anything_v2_metric_hypersim_vits.pth out_dir/

Produces out_dir/dinov2_vits14.safetensors and out_dir/da2_head_vits.safetensors.
"""

import os
import sys

import torch
from safetensors.torch import save_file

ckpt_path, out_dir = sys.argv[1], sys.argv[2]
sd = torch.load(ckpt_path, map_location="cpu")
os.makedirs(out_dir, exist_ok=True)

dino = {}
head = {}
for k, v in sd.items():
    if k.startswith("pretrained."):
        nk = k[len("pretrained."):]
        if nk == "mask_token":
            continue
        dino[nk] = v.contiguous()
    elif k.startswith("depth_head."):
        head[k] = v.contiguous()
    else:
        print("skipped", k)

# The candle dinov2 module requires an ImageNet classifier head; DA2
# checkpoints don't have one, so synthesize zeros (it is never evaluated).
embed = dino["cls_token"].shape[-1]
dino["head.weight"] = torch.zeros(1000, 2 * embed)
dino["head.bias"] = torch.zeros(1000)

save_file(dino, os.path.join(out_dir, "dinov2_vits14.safetensors"))
save_file(head, os.path.join(out_dir, "da2_head_vits.safetensors"))
print("dino keys", len(dino), "head keys", len(head), "embed", embed)
