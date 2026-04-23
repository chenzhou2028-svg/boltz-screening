#1.安装 Boltz
!pip install -q boltz[cuda]

#2.填写你的蛋白和 ligand
project_name = "ternary_screen"

protein_A = "PASTE_YOUR_PROTEIN_A_SEQUENCE"
protein_B = "PASTE_YOUR_PROTEIN_B_SEQUENCE"

ligands = [
    ("lig1", "CCO"),
    ("lig2", "CCN"),
    ("lig3", "CCC"),
]

#3.挂载 Google Drive，用来保存结果
from google.colab import drive
drive.mount('/content/drive')

#4. 一键批量运行
import os
import gc
import json
import time
import shutil
import subprocess
import pandas as pd

base_drive_dir = f"/content/drive/MyDrive/{project_name}"
yaml_dir = os.path.join(base_drive_dir, "yaml_inputs")
result_dir = os.path.join(base_drive_dir, "results")
summary_dir = os.path.join(base_drive_dir, "summaries")
log_dir = os.path.join(base_drive_dir, "logs")

for d in [base_drive_dir, yaml_dir, result_dir, summary_dir, log_dir]:
    os.makedirs(d, exist_ok=True)

def clear_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

def build_yaml(smiles):
    return f"""version: 1

sequences:
  - protein:
      id: A
      sequence: {protein_A}

  - protein:
      id: B
      sequence: {protein_B}

  - ligand:
      id: C
      smiles: "{smiles}"
"""

def find_result_files(out_dir):
    cifs, pdbs, jsons = [], [], []
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            full = os.path.join(root, f)
            if f.endswith(".cif"):
                cifs.append(full)
            elif f.endswith(".pdb"):
                pdbs.append(full)
            elif f.endswith(".json"):
                jsons.append(full)
    return cifs, pdbs, jsons

def parse_failure_reason(log_text):
    t = log_text.lower()
    if "ran out of memory" in t or "out of memory" in t:
        return "GPU_OOM"
    if "number of failed examples" in t:
        return "FAILED_EXAMPLE"
    if "traceback" in t:
        return "PYTHON_ERROR"
    return "UNKNOWN"

# 生成 YAML
for name, smiles in ligands:
    with open(os.path.join(yaml_dir, f"{name}.yaml"), "w") as f:
        f.write(build_yaml(smiles))

results = []

for name, smiles in ligands:
    print(f"\n=== Running {name} ===")
    clear_gpu()

    yaml_path = os.path.join(yaml_dir, f"{name}.yaml")
    out_dir = os.path.join(result_dir, name)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    cmd = ["boltz", "predict", yaml_path, "--out_dir", out_dir, "--use_msa_server"]

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - start, 2)

    log_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path = os.path.join(log_dir, f"{name}.log")
    with open(log_path, "w") as f:
        f.write(log_text)

    cifs, pdbs, jsons = find_result_files(out_dir)
    success = (proc.returncode == 0) and (len(cifs) > 0 or len(pdbs) > 0)

    row = {
        "ligand_name": name,
        "smiles": smiles,
        "success": success,
        "failure_reason": "" if success else parse_failure_reason(log_text),
        "elapsed_sec": elapsed,
        "cif_count": len(cifs),
        "pdb_count": len(pdbs),
        "json_count": len(jsons),
        "first_cif": cifs[0] if cifs else "",
        "first_pdb": pdbs[0] if pdbs else "",
        "first_json": jsons[0] if jsons else "",
        "log_path": log_path,
        "out_dir": out_dir,
    }

    results.append(row)
    pd.DataFrame(results).to_csv(os.path.join(summary_dir, "run_summary.csv"), index=False)

    print("Success:", row["success"])
    print("Reason :", "OK" if row["success"] else row["failure_reason"])
    print("Time   :", row["elapsed_sec"], "sec")

print("\nDone.")
print("Saved to:", base_drive_dir)

#5. 查看结果表
import pandas as pd
summary_path = f"/content/drive/MyDrive/{project_name}/summaries/run_summary.csv"
df = pd.read_csv(summary_path)
df

#6.只看成功的结构文件
df[df["success"] == True][["ligand_name", "first_cif", "first_pdb", "first_json"]]

#7.下载结果表
from google.colab import files
files.download(f"/content/drive/MyDrive/{project_name}/summaries/run_summary.csv")
