import numpy as np
import tensorflow as tf


def agnosticize(model, val=0.01, prune_ratio=(1, 4)):
    """Replace the trainable variables of model with 
    matrices containing a single shared weight.

    Args:
        model - (tensorflow.keras.Sequential) - the model to be modified.
        val - (float) - the single shared weight value. If is None, the
            original weights in the model will be preserved.
        prune_ratio - (tuple (rate of weights remaining, total weights) 
            or list of tuples) - the ratio of weights to be remained after 
            pruning. For example, (2,5) will set approximately 1-(2/5)=3/5 of 
            weights in all trainable variables to zeros. [(1,3),(1,2)] will 
            set 2/3 of weights in the first trainable variable to zeros, and 
            1/2 of weight in the second trainable variable to zeros. If 
            prune_ratio is a list of tuple, the length of list must match the
            total number of trainable_variables of the model.
    """
    if type(prune_ratio[0]) is int:
        ratios = [prune_ratio]*len(model.trainable_variables)
    else:
        ratios = prune_ratio
    assert len(ratios) == len(model.trainable_variables)

    nonzeros, zeros = 0, 0
    for i, var in enumerate(model.trainable_variables):
        ratio = ratios[i]
        w = var.numpy()
        if val is not None:
            nw = np.full_like(w, fill_value=val)
        else:
            nw = w
        mask = np.cast['int32'](np.random.randint(
            1, ratio[1]+1, w.shape) <= ratio[0])
        nw = np.multiply(mask, nw)

        nonzero = np.count_nonzero(nw)
        zero = np.prod(nw.shape)-nonzero
        print('nonzero:{} zero:{} weights remain:{:.1f}%'.format(
            nonzero, zero, (nonzero/np.prod(nw.shape))*100))
        nonzeros += nonzero
        zeros += zero
        var.assign(nw)
    print('total weights remain {:.1f}%'.format(
        (nonzeros/(nonzeros+zeros))*100))


def random_prune(model, prune_rate, input_prune_rate=0.8, skip_1d=True):
    """Randomly prune trainable_variables of model.

    Args:
        model - (tensorflow.keras.Sequential) - the model to be modified.
        prune_rate - (0.0 <= float <= 1.0) - the percentage of weights 
            remain in model after pruning.
        input_prune_rate - (0.0 <= float <= 1.0) - upper bound for the 
            prune rate for the weight matrix associated with the input layer. 
    """

    if prune_rate > input_prune_rate:
        input_prune_rate = prune_rate

    nonzeros_before = 0
    nonzeros_after = 0
    d = 1
    for i, var in enumerate(model.trainable_variables):
        nw = var.numpy()
        if skip_1d and nw.ndim == 1:
            continue

        nonzero = np.count_nonzero(nw)
        nonzeros_before += nonzero

        rate = prune_rate if i != 0 else input_prune_rate
        mask = np.cast['int32'](np.random.uniform(0, 1, nw.shape) < rate)
        nw = mask*nw

        nonzero_ = np.count_nonzero(nw)
        zero_ = np.prod(nw.shape)-nonzero_
        print("nonzero:{} zero:{} weights remain:{:.1f}%".format(
            nonzero_, zero_, (nonzero_/np.prod(nw.shape))*100))
        nonzeros_after += nonzero_

        var.assign(nw)

    print("nonzero before: {} non-zero after: {} weights remain:{:.1f}%"
          .format(nonzeros_before, nonzeros_after, (nonzeros_after/nonzeros_before)*100))
