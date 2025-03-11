import torch
import random
import numpy as np
from transformers import set_seed

def set_seed_forall(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # from transformers

    # Force deterministic CuDNN behavior:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
