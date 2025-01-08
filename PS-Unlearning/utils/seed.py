'''
Author       : Doin4
Date         : 2023-09-06 22:10:18
LastEditors  : Doin4
LastEditTime : 2023-09-06 22:11:26
Description  : Seed file.
'''
import os
import random

import torch
import numpy as np

def seed_everything(seed=301):
    """This function can fix the random seed.

    Args:
        seed (int, optional): The seed number. Defaults to 301.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    pass