# inspect_actions.py
import glob, h5py, numpy as np

files = sorted(glob.glob('datasets/*.hdf5') + glob.glob('datasets/**/*.hdf5', recursive=True))
if not files:
    print("No hdf5 found"); raise SystemExit(1)

all_actions = []
per_demo_stats = {}
for f in files:
    with h5py.File(f,'r') as h5f:
        for name in h5f['data'].keys():
            grp = h5f['data'][name]
            if 'actions' not in grp:
                continue
            a = np.array(grp['actions'], dtype=np.float32)
            if a.size == 0:
                continue
            a2 = a.reshape(-1, a.shape[-1])
            all_actions.append(a2)
            per_demo_stats[f"{f}:{name}"] = {
                'shape': a2.shape,
                'mean': a2.mean(axis=0).tolist(),
                'std': a2.std(axis=0).tolist(),
                'min': a2.min(axis=0).tolist(),
                'max': a2.max(axis=0).tolist(),
                'zero_rows_frac': float((a2==0).all(axis=1).mean())
            }

if not all_actions:
    print("No actions found"); raise SystemExit(1)

A = np.concatenate(all_actions, axis=0)
print("ALL actions shape:", A.shape)
print("per-dim mean:", A.mean(axis=0))
print("per-dim std :", A.std(axis=0))
print("overall min/max:", A.min(), A.max())
print("fraction rows all-zero:", float((A==0).all(axis=1).mean()))
print("\nSample per-demo stats (first 10):")
for i,(k,v) in enumerate(per_demo_stats.items()):
    if i>=10: break
    print(k, v)