import numpy as np

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (H,W,K) where the last dim is a one
    hot encoding vector

    """

    _mask = [mask == i+1 for i in range(num_classes)]
    _mask = np.transpose(np.array(_mask), (1, 2, 0)).astype(np.uint8)

    return _mask

def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """

    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask


def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (B, H, W) to (B, C, H, W) where the last dim is a one
    hot encoding vector

    """

    _mask = [mask == i + 1 for i in range(num_classes)]
    _mask = np.transpose(np.array(_mask), (1, 2, 0)).astype(np.uint8)

    return _mask


def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """

    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask