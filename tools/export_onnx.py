# Depth Anything V2 metric (Hypersim, small) -> ONNX with a dynamic input size (multiples of 14), for TensorRT.
#   pip install torch transformers onnx onnxscript && python tools/export_onnx.py da2_metric_hypersim_vits.onnx
import sys, torch
from transformers import AutoModelForDepthEstimation
import transformers.models.depth_anything.modeling_depth_anything as modeling

# Upstream's head casts the output size to Python ints, which freezes it at the traced size.
def head_forward(self, hidden_states, patch_height, patch_width):
    depth = self.conv1(hidden_states[self.head_in_index])
    size = (patch_height * self.patch_size, patch_width * self.patch_size)
    depth = torch.nn.functional.interpolate(depth, size, mode="bilinear", align_corners=True)
    depth = self.activation1(self.conv2(depth))
    return (self.activation2(self.conv3(depth)) * self.max_depth).squeeze(dim=1)
modeling.DepthAnythingDepthEstimationHead.forward = head_forward
out = sys.argv[1]
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf").eval()
class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(pixel_values=x).predicted_depth
patches_h, patches_w = torch.export.Dim("patches_h", min=4, max=74), torch.export.Dim("patches_w", min=4, max=74)
torch.onnx.export(Wrap(model), (torch.randn(1, 3, 364, 448),), out, input_names=["image"], output_names=["depth"],
                            dynamo=True, opset_version=18, external_data=False,
                            dynamic_shapes=({2: 14 * patches_h, 3: 14 * patches_w},))
print("wrote", out)
