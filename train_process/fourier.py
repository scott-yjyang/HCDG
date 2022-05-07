import torchvision.transforms as T
import torch
from torch.utils.data import Dataset as TorchDataset

import random
import numpy as np
from PIL import Image
from .fourier_utils import FDA_source_to_target_np, FDA_source_to_target
import scipy.misc
import math



def fourier_amplitude_mix(im_src, im_trg, L=1.0):

    sigma = random.uniform(math.sqrt(1.0/(2*math.pi)), L)
    src_in_trg, trg_in_src = FDA_source_to_target( im_src, im_trg, sigma=sigma )

    return src_in_trg, trg_in_src


