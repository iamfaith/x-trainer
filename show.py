import h5py

dataset_file = './datasets/test.hdf5'
# dataset_file = './datasets/task1-0421-1950-left5.hdf5'
with h5py.File(dataset_file, 'r') as f:
   # print('data' in f)  # 确认顶层是否有 'data'
   # print(list(f['data'].keys()))  # 列出子数据集
   for name, grp in f['data'].items():
       print(grp.attrs.get("success", None), name)
