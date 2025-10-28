import subprocess
import time
import csv
from pathlib import Path

RAW_DIR = "data/raw_pdf"
OUT_BASE = "data/out_bench/"
REPORT_BASE = "bench_report"
LIMIT = 32  # ← Ajuste ici

batch_sizes = [1, 2, 4, 8, 16]
results = []

for b in batch_sizes:
    print(f"\n🚀 Running batch_size={b}")
    out_dir = Path(f"{OUT_BASE}_{b}")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = f"{REPORT_BASE}_{b}.csv"

    start = time.perf_counter()
    subprocess.run([
        "poetry", "run", "python", "src/patent_pipeline/pipeline.py",
        "--backend", "doctr",
        "--raw_dir", RAW_DIR,
        "--out_dir", str(out_dir),
        "--report_file", report_file,
        "--batch_size", str(b),
        "--limit", str(LIMIT),
        "--force"
    ], check=True)
    elapsed = time.perf_counter() - start
    print(f"✅ Batch {b}: {elapsed:.1f}s total, {elapsed / LIMIT:.2f}s/doc")
    results.append((b, elapsed, elapsed / LIMIT))

# Save benchmark results
with open("batch_benchmark.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["batch_size", "total_time_s", "time_per_doc_s"])
    writer.writerows(results)

print("\n🏁 Benchmark complete! Results saved to batch_benchmark.csv")
