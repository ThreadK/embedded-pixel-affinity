import numpy as np
import itertools
import torch

class TestAugmentor(object):
    """Test-time augmentor. 
    
    Our test-time augmentation includes horizontal/vertical flips 
    over the `xy`-plane, swap of `x` and `y` axes, and flip in `z`-dimension, 
    resulting in 16 variants. Considering inference efficiency, we also 
    provide the option to apply only `x-y` swap and `z`-flip, resulting in 4 varia