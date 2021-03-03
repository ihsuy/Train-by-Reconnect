## Train by Reconnect: Decoupling Locations of Weights from Their Values

This repository contains the official code for the NeurIPS 2020 paper *Train by Reconnect: Decoupling Locations of Weights from Their Values* by Yushi Qiu and Reiji Suda.

<p align="left" style="float: left;">
  <img src="https://github.com/ihsuy/Train-by-Reconnect/blob/main/Images/perm4.gif?raw=true" height="230">
</p> 


> **Train by Reconnect: Decoupling Locations of Weights from Their Values**<br>
> Yushi Qiu and Reiji Suda <br>
> The University of Tokyo
>
> **Abstract:** What makes untrained deep neural networks (DNNs) different from the trained performant ones? By zooming into the weights in well-trained DNNs, we found that it is the *location* of weights that holds most of the information encoded by the training. Motivated by this observation, we hypothesized that weights in DNNs trained using stochastic gradient-based methods can be separated into two dimensions: the location of weights, and their exact values. To assess our hypothesis, we propose a novel method called *lookahead permutation* (LaPerm) to train DNNs by reconnecting the weights. We empirically demonstrate LaPerm's versatility while producing extensive evidence to support our hypothesis: when the initial weights are random and dense, our method demonstrates speed and performance similar to or better than that of regular optimizers, e.g., *Adam*. When the initial weights are random and sparse (many zeros), our method changes the way neurons connect, achieving accuracy comparable to that of a well-trained dense network. When the initial weights share a single value, our method finds a weight agnostic neural network with far-better-than-chance accuracy.
>


## Dependencies
Code in this repository requires:
- Python 3.6 or higher
- Tensorflow v2.1.0 or higher
and the requirements highlighted in [requirements.txt](./requirements.txt)

## Table of Contents
This repository contains the following contents:
- **train_by_reconnect**: minimum code for reproducing main results mentioned in the paper. The code is commented and accompanied with working examples in [notebooks](./notebooks).
    - [LaPerm.py](./train_by_reconnect/LaPerm.py)
        - `LaPerm`: [Tensorflow](https://www.tensorflow.org/) implementation of LaPerm (Section 4).
        - `LaPermTrainLoop`: A custom train loop that applies LaPerm to [tensorflow.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model).
    - [weight_utils.py](./train_by_reconnect/weight_utils.py)
        - `agnosticize`: Replace the weights in a model with a single shared value. (Section 5.5)
        - `random_prune`: Randomly prune the model. (Section 5.4)
    - [viz_utiles.py](./train_by_reconnect/viz_utils.py)
        - `Profiler`: Plot weight profiles for a given model. (Section 2)
        - `PermutationTracer`: Visualize and trace how the locations of weights has changed.       
- **notebooks**: [Jupyter-notebooks](./notebooks) containing the model definitions and experiment configurations for reconducting or extending the experiments (training + evaluation). Detailed instructions can be found inside the notebooks.
    - [`Conv2.ipynb`](./notebooks/Conv2.ipynb), [`Conv4.ipynb`](./notebooks/Conv4.ipynb), [`Conv13.ipynb`](./notebooks/Conv13.ipynb), [`Conv7.ipynb`](./notebooks/Conv7.ipynb), [`ResNet50.ipynb`]((./notebooks/ResNet50.ipynb)): For experiments mentioned in Section 5.1~5.4.
    - [`F1_and_F2.ipynb`](./notebooks/F1_and_F2.ipynb): For experiments mentioned in Section 5.5.
    - [`Weight_profiles.ipynb`](./notebooks/Weight_profiles.ipynb): For visualizations mentioned in Section 2.
- **pretrain**: pre-train weights for main results mentioned in the paper. (For detailed model definitions, please refer to 'notebooks`)
    | Models     | Top-1 | *p%* | *k* | Dataset | Section | Weights |
    | ---------- |:-----:| ----:| ---:| -------:| -------:| -----------:| 
    | [Conv7](./pretrained/Conv7.h5)      | 99.72%| 0%   | 1200|   MNIST |     5.1 | He Uniform  |
    | [Conv2](./pretrained/Conv2.h5)      | 78.21%| 0%   | 1000| CIFAR-10|5.2, 5.4 | He Uniform  |
    | [Conv4](./pretrained/Conv4.h5)      | 89.17%| 0%   | 1000| CIFAR-10|5.2, 5.4 | He Uniform  |
    | [Conv13](./pretrained/Conv13.h5)     | 92.21%| 0%   | 1000| CIFAR-10|5.2, 5.4 | He Uniform  |
    | [ResNet50](./pretrained/resnet50.h5)   | 92.53%| 0%   |  400| CIFAR-10|     5.4 | He Uniform  |
    | [ResNet50](./pretrained/resnet50_30.h5)   | 92.32%| 30%  |  800| CIFAR-10|     5.4 | He Uniform  |
    | [ResNet50](./pretrained/resnet50_50.h5)   | 92.02%| 50%  |  800| CIFAR-10|     5.4 | He Uniform  |
    | [ResNet50](./pretrained/resnet50_70.h5)   | 90.97%| 70%  |  800| CIFAR-10|     5.4 | He Uniform  |
    | [F1](./pretrained/F1.h5)         | 85.46%| 40%  |  250|   MNIST |     5.5 | Shared 0.08 |
    | [F2](./pretrained/F2.h5)         | 78.14%| 92%  |  250|   MNIST |     5.5 | Shared 0.03 |
    
    - ***p%***: Percentage of weights that are randomly pruned before training, e.g., *p*=10% meaning 90% of weights are remained non-zero. (Section 5.4)

    - ***k***: Sync period used to perform the experiment. (Section 4)
    - ***Weights***: Mechanism used to generate the random weights.
        - He Uniform: [He et al. 2015](https://arxiv.org/abs/1502.01852)
        - Shared 0.08: the weights are sampled from the set {0, 0.08}.
        - Shared 0.03: the weights are sampled from the set {0, 0.03}.
    - Datasets: [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).


## Steps to load the pre-trained weights
1. Locate the weight's corresponding jupyter-notebook in [notebooks](./notebooks). For example, for the weight named `Conv7.h5`, please look for [Conv7.ipynb](./notebooks/Conv7.ipynb) for the model definition and experiment configurations.
2. Define the `model` as demonstrated in the notebook.
3. Load the weights to `model` by
    ```python
    model.load_weights('../pretrained/Conv7.h5')
    ```
---
## Resources

All material related to our paper is available via the following links:

| Resources                    | Link
| :--------------           | :----------
| Paper PDF | https://arxiv.org/abs/2003.02570
| Project page | TBA
| Notebooks to reproduce experiments | [Link Notebooks](./notebooks)
| Source code | [Link Github](https://github.com/ihsuy/Train-by-Reconnect)
| Summary video | TBA
| Presentation slides | TBA
| Poster | [Link](https://github.com/ihsuy/Train-by-Reconnect/blob/main/NeurIPS%20Poster.pdf)

---
## License
MIT
