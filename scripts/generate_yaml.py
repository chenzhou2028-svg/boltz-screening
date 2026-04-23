import json
import pandas as pd
import os

# 读取配置
with open("configs/proteins.json") as f:
    proteins = json.load(f)

ligands = pd.read_csv("configs/ligands.csv")

os.makedirs("yaml_inputs", exist_ok=True)

for _, row in ligands.iterrows():
    name = row["ligand_name"]
    smiles = row["smiles"]

    yaml_text = f"""version: 1

sequences:
  - protein:
      id: A
      sequence: {proteins["protein_A_sequence"]}

  - protein:
      id: B
      sequence: {proteins["protein_B_sequence"]}

  - ligand:
      id: C
      smiles: "{smiles}"
"""

    with open(f"yaml_inputs/{name}.yaml", "w") as f:
        f.write(yaml_text)

print("YAML files generated.")
