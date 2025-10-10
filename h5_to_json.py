# h5_to_json.py
import os, json, numpy as np, pandas as pd
from datetime import datetime

def export_h5_dir_to_json(keypoints_dir: str, out_json: str, day: str=None):
    """
    Crea lsa_samples.json con schema:
    { "t":15, "d":126, "by_date": { "YYYY-MM-DD":[ {label, seq:[...]} ] } }
    """
    by_date = {}
    t = None; d = 126
    day = day or datetime.now().strftime("%Y-%m-%d")

    for fn in sorted(os.listdir(keypoints_dir)):
        if not fn.endswith(".h5"): continue
        label = os.path.splitext(fn)[0]
        df = pd.read_hdf(os.path.join(keypoints_dir, fn), key='data')
        for _, g in df.groupby(df['sample'] if 'sample' in df.columns else 0):
            seq = np.stack([np.asarray(row['keypoints'], dtype=float) for _, row in g.iterrows()], axis=0)
            t = seq.shape[0]
            by_date.setdefault(day, []).append({"label": label, "seq": seq.tolist()})

    root = { "t": t or 15, "d": d, "by_date": by_date }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(root, f)
    print("Escrito:", os.path.abspath(out_json))

# ejemplo:
# export_h5_dir_to_json("data/keypoints", "lsa_samples.json")
