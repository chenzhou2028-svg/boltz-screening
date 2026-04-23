import os
import pandas as pd

records = []

for root, dirs, files in os.walk("results"):
    for f in files:
        if f.endswith(".cif") or f.endswith(".pdb"):
            records.append({
                "file": f,
                "path": os.path.join(root, f)
            })

df = pd.DataFrame(records)
df.to_csv("summaries/structures.csv", index=False)

print("Summary saved.")
