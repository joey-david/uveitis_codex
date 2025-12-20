import os
import time
import csv
import torch

from src.mtsn.mtsn_model import MTSN
from config import load_config


def benchmark(batch_sizes=(8, 16, 32), iters=50):
    cfg = load_config()
    device = torch.device(cfg.base.device if torch.cuda.is_available() else "cpu")
    model = MTSN().to(device)
    model.eval()

    H, W = cfg.mtsn.transforms.resize
    results = []

    # warmup
    with torch.no_grad():
        x = torch.randn(4, 3, H, W, device=device)
        for _ in range(10):
            _ = model.encode(x)

    for bs in batch_sizes:
        x = torch.randn(bs, 3, H, W, device=device)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t0 = time.time()
        with torch.no_grad():
            for _ in range(iters):
                _ = model.encode(x)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        dt = time.time() - t0
        images = bs * iters
        ips = images / dt
        results.append((bs, ips, dt))
        print(f"bs={bs}: {ips:.1f} img/s, time={dt:.3f}s")

    # persist
    os.makedirs("metrics", exist_ok=True)
    out = os.path.join("metrics", "mtsn_bench.csv")
    header = ("batch_size", "images_per_s", "total_time_s")
    write_header = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for bs, ips, dt in results:
            w.writerow([bs, f"{ips:.3f}", f"{dt:.6f}"])


if __name__ == "__main__":
    benchmark()

