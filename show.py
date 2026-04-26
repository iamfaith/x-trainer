import h5py

dataset_file = './datasets/merge_50.hdf5'
# dataset_file = './datasets/task1-0421-1950-left5.hdf5'
with h5py.File(dataset_file, 'r') as f:
   # print('data' in f)  # 确认顶层是否有 'data'
   # print(list(f['data'].keys()))  # 列出子数据集
   for name, grp in f['data'].items():
       print(grp.attrs.get("success", None), name)


import glob, h5py, numpy as np
files = sorted(glob.glob('datasets/*.hdf5') + glob.glob('datasets/**/*.hdf5', recursive=True))
A_list=[]
for f in files:
    with h5py.File(f,'r') as h:
        for name in h['data'].keys():
            grp = h['data'][name]
            if 'actions' in grp:
                a = np.array(grp['actions'], dtype=np.float32)
                if a.size>0:
                    A_list.append(a.reshape(-1,a.shape[-1]))
if not A_list:
    print('no actions'); raise SystemExit
A = np.concatenate(A_list, axis=0)
print('shape', A.shape)
print('per-dim mean', np.mean(A,0))
print('per-dim std ', np.std(A,0))
print('min/max', A.min(), A.max())
print('fraction rows all-zero', np.mean((A==0).all(axis=1)))