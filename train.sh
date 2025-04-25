docker run --rm --runtime nvidia \
    --gpus all \
    --shm-size=16G \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v $(pwd):/workspace \
    -it joyvasa \
    bash
    # python train.py
    # python -c 'import torch; print(torch.cuda.is_available())'  # train.py
