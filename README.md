As part of my interviews for a PhD position at the AutoML group located in
Hannover at the Leibniz University, I was tasked the following coding challenge

## Objective: Find the optimal learning rate of a neural network

* Optimizer: Implement Bayesian Optimization (BO) as a global optimization
    approach use a Gaussian Process as a predictive model (you can use any
    existing GP library) use expected improvement (EI) as an acquisition
    function. Use an optimization budget of 10 function evaluations if you
    want to watch some intro videos for BO, you can find our MOOC on AutoML
    on the ai-campus.org For the deep neural network, please follow the
    specifications below: A small ResNet, e.g., a ResNet-9 --- should not take
    too long for training, but should achieve a reasonable performance

* ResNet optimization: SGD

* Deep learning framework: PyTorch

* Dataset: KMNIST (https://github.com/rois-codh/kmnist)
    ![https://github.com/rois-codh/kmnist](./Plots/kmnist_examples.png)

* Plotting: Starting with the second iteration of Bayesian Optimization, 
    plot all
    observations, the posterior mean, uncertainty estimate and the acquisition
    function after each iteration See here for an exemplary plots (on a 
    different task):
    https://towardsdatascience.com/shallow-understanding-on-bayesian-optimization-324b6c1f7083

* Programming language:
    Python
  
    PEP8 (found here: https://www.python.org/dev/peps/pep-0008/)
  
    Doc-Strings

## My Solution

### ResNet & ResdiualBlocks
Following the formulation of https://github.com/matthias-wright/cifar10-resnet,
I interpreted a ResNet-9 as a 9-layer ResNet from this paper:

    @inproceedings{he2016deep,
      title={Deep residual learning for image recognition},
      author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
      booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages={770--778},
      year={2016}
    }

Only alteration to this paper: instead of avg. pooling, a maxpooling 2x2 
with stride 2 is used.

Notice, the ResidualBlock class implements both the dashed & solid line 
identity and 1x1 convolution projections in a single and adaptive class.
Further the interface easily allows to scale the ResidualBlock to various 
channel configurations and an arbitrary skipping distance.
This allows the ResNet class to be a sequential composite of ResidualBlocks,
which facilitates the entire class.

## Bayesian Optimisation

Be well aware, that the pyro gp implementation apparently can predict zero 
& negative variances. To avoid this, such values are explicitly set to 1e-10.
