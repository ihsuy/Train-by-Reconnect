import textwrap
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


def Profiler(model, nrows=None, ncols=None, skip_1d=True, path=None, wrapwidth=30):
    """Plot weight profiles for trainable_variables of model.

    Args:
        model - (tensorflow.keras.Sequential) - the model to be plotted.
        nrows - (int) - number of rows 
        ncols - (int) - number of columns
        skip_1d - (boolean) - whether to skip trainable_variable
            with number of dimension equals to 1, e.g., biases.
        path - (str) - save plotted image to path.
        wrapwidth - (int) - width for textwrap.wrap.
    """
    w = [var.numpy() for var in model.trainable_variables]
    names = [var.name for var in model.trainable_variables]
    plottable = []
    plot_names = []
    dim_lim = 0 if not skip_1d else 1
    for i, item in enumerate(w):
        if item.ndim > dim_lim and item.shape[-1] > dim_lim:
            plottable.append(item)
            plot_names.append(names[i])
    n = len(plottable)

    if nrows is None or ncols is None:
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n/ncols)

    print("Plotting {} items\nUsing grid of size {} x {}".format(n, nrows, ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols*3*1.5, nrows*2*1.5))
    for r in range(nrows):
        for c in range(ncols):
            index = r*ncols+c
            if index >= n:
                if nrows == 1:
                    if ncols == 1:
                        axes.set_axis_off()
                    else:
                        axes[c].set_axis_off()
                else:
                    axes[r][c].set_axis_off()
                continue

            data = plottable[index]
            ndim = data.ndim
            if ndim == 4:
                data = data.reshape((np.prod(data.shape[:3]), data.shape[3]))

            data = np.sort(data, axis=0)

            title = plot_names[index]+" {}".format(data.shape)
            title = '\n'.join(textwrap.wrap(title, wrapwidth))
            if nrows == 1:
                if ncols == 1:
                    axes.plot(data)
                    axes.set_title(title)
                else:
                    axes[c].plot(data)
                    axes[c].set_title(title)
            else:
                axes[r][c].plot(data)
                axes[r][c].set_title(title)

    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path, format='png')


def PermutationTracer(A, B, figsize=(15, 15), arrow_alpha=1, max_rad=0.2, on_self=False,
                      diff_only=True, cmap=None):
    """ Given 2 matrices A and B, where B can be obtained by permuting the entries of A
    This method connects pixels of the same values between two matrixes.

    Args:
        arrow_alpha - (float) Transparency of arrows
        max_rad - (float) Maximum arrow curvature
        on_self - (boolean) - Whether show permutation on the left hand side image only.
        diff_only - (boolean) - whether connects pixels that changed in locations.
    """

    shapeA = A.shape
    shapeB = B.shape

    assert shapeA == shapeB, "A and B must have the same shapes."

    A = A.ravel()
    B = B.ravel()

    ranker, locator, mapper = {}, {}, {}
    argB = np.argsort(B)
    rankB = np.argsort(argB)
    for val, rank in zip(B, rankB):
        ranker[val] = rank
    for loc, arg in enumerate(argB):
        locator[loc] = arg
    for i in range(len(A)):
        mapper[i] = locator[ranker[A[i]]]

    A = A.reshape(shapeA)
    B = B.reshape(shapeB)

    # Plot
    fig = plt.figure(figsize=figsize)
    if not on_self:
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(224)
    else:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    ax1.matshow(A, cmap=cmap)
    ax2.matshow(B, cmap=cmap)
    ax1.axis('off')
    ax2.axis('off')

    # Connect pixels
    for i in range(shapeA[0]):
        for j in range(shapeA[1]):

            index = i*shapeA[1] + j
            indexB = mapper[index]

            if diff_only and index == indexB:
                continue

            xyA = (indexB % shapeA[1], indexB//shapeA[1])
            xyB = (j, i)

            axesA = ax2
            axesB = ax1
            if on_self:
                axesB = ax1
                axesA = ax1

            con = ConnectionPatch(xyA=xyA, xyB=xyB,
                                  coordsA="data", coordsB="data",
                                  axesA=axesA, axesB=axesB,
                                  color='turquoise' if np.random.randint(
                                      2) else 'darkorange',
                                  linewidth=2,
                                  arrowstyle='<-',
                                  connectionstyle="arc3,rad={}".format(
                                      np.random.uniform(-max_rad, max_rad)),
                                  alpha=arrow_alpha)
            if on_self:
                ax1.add_artist(con)
            else:
                ax2.add_artist(con)
