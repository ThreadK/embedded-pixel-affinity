import cv2
import math
import numpy as np
from .augmentor import DataAugment

class MisAlignment(DataAugment):
    """Mis-alignment data augmentation of image stacks.
    
    Args:
        displacement (int): maximum pixel displacement in `xy`-plane. Default: 16
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self, 
                 displacement=16, 
                 rotate_ratio=0.0,
                 p=0.5):
        super(MisAlignment, self).__init__(p=p)
        self.displacement = displacement
        self.rotate_ratio = rotate_ratio
        self.set_params()

    def set_params(self):
        self.sample_params['add'] = [0, 
                                     int(math.ceil(self.displacement / 2.0)), 
                                     int(math.ceil(self.displacement / 2.0))]

    def misalignment(self, data, random_state):
        images = data['image'].copy()
        labels = data['label'].copy()

        out_shape = (images.shape[0], 
                     images.shape[1]-self.displacement, 
                     images.shape[2]-self.displacement)    
        new_images = np.zeros(out_shape, images.dtype)
        new_labels = np.zeros(out_shape, labels.dtype)

        x0 = random_state.randint(self.displacement)
        y0 = random_state.randint(self.displacement)
        x1 = random_state.randint(self.displacement)
        y1 = random_state.randint(self.displacement)
        idx = random_state.choice(np.array(range(1, out_shape[0]-1)), 1)[0]

        if random_state.rand() < 0.5:
            # slip misalignment
            new_images = images[:, y0:y0+out_shape[1], x0:x0+out_shape[2]]
            new_labels = labels[:, y0:y0+out_shape[1], x0:x0+out_shape[2]]
            new_images[idx] = images[idx, y1:y1+out_shape[1], x1:x1+out_shape[2]]
            new_labels[idx] = labels[idx, y1:y1+out_shape[1], x1:x1+out_shape[2]]
        else:
            # translation misalignment
            new_images[:idx] = images[:idx, y0:y0+out_shape[1], x0:x0+out_shape[2]]
            new_labels[:idx] = labels[:idx, y0:y0+out_shape[1], x0:x0+out_shape[2]]
            new_images[idx:] = images[idx:, y1:y1+out_shape[1], x1:x1+out_shape[2]]
            new_labels[idx:] = labels[idx:, y1:y1+out_shape[1], x1:x1+out_shape[2]]
    
        return new_images, new_labels

    def misalignment_rotate(self, data, random_state):
        images = data['image'].copy()
        labels = data['label'].copy()

        height, width = images.shape[-2:]
        assert height == width
        M = self.random_rotate_matrix(height, random_state)
        idx = random_state.choice(np.array(range(1, images.shape[0]-1)), 1)[0]

        if random_state.rand() < 0.5:
            # slip misalignment
            images[idx] = cv2.warpAffine(images[idx], M, (height,width), 1.0, 
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            labels[idx] = cv2.warpAffine(labels[idx], M, (height,width), 1.0, 
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        else:
            # translation misalignment
            for i in range(idx, images.shape[0]):
                images[i] = cv2.warpAffine(images[i], M, (height,width), 1.0, 
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                labels[i] = cv2.warpAffine(labels[i], M, (height,width), 1.0, 
               