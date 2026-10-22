"""Exercise CUDA matmul and gradients, with CPU reference; no model inference."""
import json
import platform
from pathlib import Path
import torch

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.cuda.is_available(), "CUDA unavailable"
x = torch.randn(64, 32, dtype=torch.float64, device="cuda", requires_grad=True)
w = torch.randn(32, 16, dtype=torch.float64, device="cuda", requires_grad=True)
out = x @ w
loss = out.square().mean()
loss.backward()
torch.cuda.synchronize()
reference = x.detach().cpu() @ w.detach().cpu()
error = (out.detach().cpu()-reference).abs().max().item()
assert error < 1e-10 and torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
record = {
    "test": "CUDA float64 matmul and backward, not HaMeR",
    "python": platform.python_version(), "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(0),
    "max_cpu_gpu_abs_error": error, "passed": True,
}
path = Path("artifacts/bootstrap/gpu_smoke.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
