# eval_model_vs_gt.py
import glob, h5py, numpy as np, torch
from lerobot.policies.factory import get_policy_class
from lerobot.scripts.server.helpers import raw_observation_to_observation

# USER: 修改下面路径
checkpoint_dir = "outputs/train/act_task1_50_chunk900/checkpoints/last/pretrained_model"
hdf5_example = "datasets/merge_50.hdf5"  # 或你的任意 hdf5

# load policy (eval mode)
PolicyClass = get_policy_class('act')
policy = PolicyClass.from_pretrained(checkpoint_dir)
policy.to('cpu')
policy.eval()

def make_raw_obs_from_group(grp):
    # 根据你的 conversion 中的键名组织 RawObservation 字典
    # 必要时修改键名/格式以匹配训练时的 raw_observation 格式
    raw = {}
    # 合并左右关节位置为 state array [16]
    left = np.array(grp['obs/left_joint_pos_rel'])
    right = np.array(grp['obs/right_joint_pos_rel'])
    state = np.concatenate([left, right], axis=1)  # (T, 16)
    # images: keep one frame (可能需要转 uint8)
    top = np.array(grp['obs/top'])
    left_img = np.array(grp['obs/left_wrist'])
    right_img = np.array(grp['obs/right_wrist'])
    # This is an example: adapt to the exact raw format expected by raw_observation_to_observation
    raw['observation.state'] = state[0]  # single step
    raw['observation.images.top'] = top[0]
    raw['observation.images.left_wrist'] = left_img[0]
    raw['observation.images.right_wrist'] = right_img[0]
    return raw, state, np.array(grp['actions'])

with h5py.File(hdf5_example,'r') as h5f:
    name = list(h5f['data'].keys())[0]
    grp = h5f['data'][name]
    raw, state_seq, actions = make_raw_obs_from_group(grp)

# convert raw -> model input (use policy.config.image_features and policy.config.lerobot_features if needed)
# NOTE: you must provide lerobot_features used at training time; here we try to use policy.config if present
lerobot_features = getattr(policy.config, 'lerobot_features', None)
image_features = getattr(policy.config, 'image_features', None)
if lerobot_features is None or image_features is None:
    print("Warning: unable to find dataset features in policy.config; ensure correct preprocessing.")
obs = raw_observation_to_observation(raw, lerobot_features or {}, image_features or {}, 'cpu')

with torch.no_grad():
    pred_chunk = policy.predict_action_chunk(obs)  # shape (B?, chunk, action_dim)
    # adapt depending on policy implementation
    pred = pred_chunk.cpu().numpy()
    print("pred shape", pred.shape)

# compare first predicted action to ground truth first action
print("GT action (first):", actions[0])
if pred.size>0:
    print("Pred action (first):", pred.ravel()[:actions.shape[1]])
# If you want MSE on many frames, iterate multiple frames building obs per frame similarly.