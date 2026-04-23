import os
import subprocess
import time
import gc
import pandas as pd

def clear_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except:
        pass

yaml_dir = "yaml_inputs"
result_dir = "results"
log_dir = "logs"

os.makedirs(result_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

results = []

for file in os.listdir(yaml_dir):
    if not file.endswith(".yaml"):
        continue

    name = file.replace(".yaml", "")
    yaml_path = os.path.join(yaml_dir, file)
    out_dir = os.path.join(result_dir, name)

    print(f"\n=== Running {name} ===")

    clear_gpu()

    cmd = [
        "boltz", "predict",
        yaml_path,
        "--out_dir", out_dir,
        "--use_msa_server"
    ]

    start = time.time()

    process = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = round(time.time() - start, 2)

    log_path = os.path.join(log_dir, f"{name}.log")
    with open(log_path, "w") as f:
        f.write(process.stdout + "\n" + process.stderr)

    success = "Number of failed examples: 0" in process.stdout

    results.append({
        "ligand": name,
        "success": success,
        "time_sec": elapsed,
        "log": log_path
    })

    pd.DataFrame(results).to_csv("summaries/run_summary.csv", index=False)

print("\nAll done.")
