FROM gdpi_isaac-lab
# 或使用你已配好的 lerobot 环境镜像

WORKDIR /workspace

# RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install lerobot -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY ./lerobot-0.3.3 /workspace/lerobot

RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install /workspace/lerobot


# 暴露推理端口
EXPOSE 5555



RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install -U protobuf grpcio  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# wget  "https://download.pytorch.org/models/resnet18-f37072fd.pth" 
COPY ./resnet18-f37072fd.pth /root/.cache/torch/hub/checkpoints/

# 将本地训练好的权重拷贝到镜像内固定路径
COPY ./pretrained_model /workspace/checkpoints/pretrained_model

# 启动推理服务，监听 0.0.0.0
CMD ["/workspace/isaaclab/_isaac_sim/python.sh", "-m", "lerobot.scripts.server.policy_server", "--host=0.0.0.0", "--port=5555", "--fps=30"]