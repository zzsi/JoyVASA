docker run --rm --runtime nvidia \
    --gpus all \
    --shm-size=16G \
    -v $(pwd):/workspace \
    -it joyvasa \
    bash
    # python train.py
    # python -c 'import torch; print(torch.cuda.is_available())'  # train.py
