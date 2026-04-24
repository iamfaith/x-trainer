
sudo apt update
sudo apt install -y ffmpeg


sudo mount -o remount,size=8G /dev/shm


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
  --dataset.root="/root/.cache/huggingface/lerobot/dstx123/task1" \
  --output_dir=outputs/train/act_dstx123_task1 \
  --policy.device=cuda \
  --num_workers=0 \
  --policy.push_to_hub=false



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
    --policy_action_horizon=16 \
    --policy_language_instruction="Grab cube and place into plate" \
    --device=cuda \
    --enable_cameras \
    --policy_checkpoint_path="/mnt/x-trainer/outputs/train/act_dstx123_task1/checkpoints/last/pretrained_model"




    --policy_checkpoint_path="./checkpoints/last/pretrained_model"
