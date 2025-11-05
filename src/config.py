# Configuration for training
config = {
    "block_size": 128,
    "batch_size": 32,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
    "max_iters": 5000,
    "lr": 3e-4,
    "dropout": 0.1,
    "device": "cuda"  # or 'cpu'
}
