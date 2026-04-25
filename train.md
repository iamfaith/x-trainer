
sudo apt update
sudo apt install -y ffmpeg


sudo mount -o remount,size=8G /dev/shm

没有num_workers 会爆共享内容

python -m lerobot.scripts.train \
  --policy.type=act \
  --dataset.repo_id=lerobot_task1 \
  --dataset.root="/root/.cache/huggingface/lerobot/dstx123/task1" \
  --output_dir=outputs/train/act_dstx123_task1 \
  --policy.device=cuda \
  --policy.push_to_hub=false


python -m lerobot.scripts.train \
  --policy.type=act \
  --dataset.repo_id=lerobot_task1 \
  --dataset.root="/root/.cache/huggingface/lerobot/merge_task1_50" \
  --output_dir=outputs/train/act_task1_50_chunk900 \
  --policy.device=cuda \
  --num_workers=0 \
  --policy.push_to_hub=false \
  --batch_size=16 \
  --steps=300000   --save_freq=2000 \
  --policy.chunk_size=900 --policy.n_action_steps=200


恢复训练

python -m lerobot.scripts.train \
  --config_path=outputs/train/act_faith_task1/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=300000   --save_freq=2000



训练结束后，启动server：

python -m pip install --upgrade grpcio  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple 

python -m lerobot.scripts.server.policy_server \
  --host=127.0.0.1 \
  --port=5555 \
  --fps=30




pip install msgpack pydantic

python scripts/evaluation/policy_inference.py \
    --task=task1 \
    --eval_rounds=10 \
    --policy_type=xtrainer_act \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=900 \
    --policy_language_instruction="Grab cube and place into plate" \
    --device=cuda \
    --enable_cameras \
    --episode_length_s 5 \
    --policy_checkpoint_path="outputs/train/act_task1/checkpoints/last/pretrained_model"



python scripts/evaluation/policy_scoring.py \
    --task=task1 \
    --eval_rounds=10 \
    --policy_type=xtrainer_act \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Grab cube and place into plate" \
    --device=cuda \
    --enable_cameras \
    --policy_checkpoint_path="outputs/train/act_task1/checkpoints/last/pretrained_model"


lerobot policy_server.py

self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)



dict_keys(['left_joint_pos_rel', 'left_joint_vel_rel', 'left_joint_pos_target', 'right_joint_pos_rel', 'right_joint_vel_rel', 'right_joint_pos_target', 'actions', 'left_wrist', 'right_wrist', 'top', 'task_description'])



['left_wrist', 'right_wrist', 'top', 'stereo_left', 'stereo_right']


lerobot-0.3.3: lerobot-0.3.3\src\lerobot\scripts\server\policy_server.py：  _obs_sanity_checks (return True)



训练（覆盖 config）：
--policy.chunk_size=900 --policy.n_action_steps=200
评估/客户端：
--policy_action_horizon=200 （等价于把 client/server 的 actions_per_chunk 设为 200）

900 步（30s @ 30fps）非常大

计划：解释每个参数含义 → 如何换算秒到步数 → 风险与建议。

- **`--policy.chunk_size`**: 模型在一次前向中预测的最大动作序列长度（单位：环境步/帧）。等同于 ACT 配置里的 `chunk_size`，在 configuration_act.py 中定义。  
- **`--policy.n_action_steps`**: 每次将模型预测的动作实际在环境中执行的步数（单位：步）。必须满足 `n_action_steps <= chunk_size`。模型可能预测一个更长的 chunk，但只执行前 `n_action_steps`，剩余丢弃或重叠使用。  
- **`--policy_action_horizon`**: 评估/客户端层面请求或期望的动作长度（相当于 client/server 的 `actions_per_chunk`）。在评估脚本/客户端中用于控制每次请求动作的长度（单位：步）。  
- **`--episode_length_s`**: 评估时单个 episode 的长度（单位：秒），用来构造仿真/评估的总时长。不是模型内部的时间步数参数；将秒转换为步需用 FPS（见下）。

换算：
- steps = seconds * fps。  
  例如你用 `fps=30`：30 秒 → `30 * 30 = 900` 步。  
  所以要实现“30s 的动作长度”，把 `--policy.chunk_size=900` 和 `--policy.n_action_steps=900`（或按策略选择较小的 `n_action_steps`）并在评估时用 `--policy_action_horizon=900`。

风险与建议：
- 内存/延迟：chunk_size 很大会显著增加 GPU 显存与推理延时，可能无法训练或部署。  
- 精度退化：长 horizon 会累计误差，通常更稳妥的做法是增大 `chunk_size` 但保守设置 `n_action_steps`（例如预测长序列但每次只执行较短段，或使用重叠预测/temporal ensembling）。  
- 数据要求：训练数据必须包含足够长的序列（帧数 ≥ 你设的 chunk_size），否则目标/标签不匹配。  
- 实验流程：先渐进增加（例如 100 → 200 → 400 → …），观察显存和行为，再尝试 900。

示例（30s @ 30fps = 900 步）：
- 训练命令（直接通过 CLI 覆盖 config）：
  `--policy.chunk_size=900 --policy.n_action_steps=900`
- 评估/客户端：
  `--policy_action_horizon=900 --episode_length_s=30`  （或 `--episode_length_s=5` 若只跑 5 秒的评估）

如需，我可以：
- 帮你把训练命令/配置改成 900 并检查 checkpoint 的实际 `chunk_size`/`n_action_steps`；或  
- 给出更保守的建议配置（例如 `chunk_size=900, n_action_steps=200`）供你逐步试验。你想先哪种？