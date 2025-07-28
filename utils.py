import random
import torch
import numpy as np  
def set_seed(seed):
    torch.manual_seed(seed)           # CPU seed
    torch.cuda.manual_seed(seed)      # GPU seed for current device
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    
    # (可选) 设置 cuDNN 的确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def softmax_with_temperature(logits, temperature=1.0):
    logits = np.array(logits) / temperature
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)
def score_ditc2softmax(scores_dict):
    return {k: softmax_with_temperature(list(scores_dict.values()))[i].item() for i, k in enumerate(scores_dict)}
