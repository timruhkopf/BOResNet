As part of my interviews for a PhD position at the AutoML group located in
Hannover at the Leibniz University, I was tasked the following coding challenge

## Objective: Find the optimal learning rate of a neural network

* Optimizer: Implement Bayesian Optimization (BO) as a global optimization
  approach use a Gaussian Process as a predictive model (you can use any
  existing GP library) use expected improvement (EI) as an acquisition
  function. Use an optimization budget of 10 function evaluations if you want
  to watch some intro videos for BO, you can find our MOOC on AutoML on the
  ai-campus.org For the deep neural network, please follow the specifications
  below: A small ResNet, e.g., a ResNet-9 --- should not take too long for
  training, but should achieve a reasonable performance

* ResNet optimization: SGD

* Deep learning framework: PyTorch

* Dataset: KMNIST (https://github.com/rois-codh/kmnist)
  ![https://github.com/rois-codh/kmnist](./Plots/kmnist_examples.png)

* Plotting: Starting with the second iteration of Bayesian Optimization, plot
  all observations, the posterior mean, uncertainty estimate and the
  acquisition function after each iteration See here for an exemplary plots (on
  a different task):
  https://towardsdatascience.com/shallow-understanding-on-bayesian-optimization-324b6c1f7083

* Programming language:
  Python

  PEP8 (found here: https://www.python.org/dev/peps/pep-0008/)

  Doc-Strings

## Proposed Solution

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

Only alteration to this paper: instead of avg. pooling, a maxpooling 2x2 with
stride 2 is used.

Notice, the ResidualBlock class implements both the dashed & solid line
identity and 1x1 convolution projections in a single and adaptive class. These
two types implement skip connections. The solid line is a simple forwarding of
the conv. block's input to the output. The dashed line adjusts the input's size
to compensate the output's difference in channels caused by the intermediate
convolutions. This is done via a learned linear projection. Further the
interface easily allows to scale the ResidualBlock to various channel
configurations and an arbitrary skipping distance. The scalability is achieved
by a nn.ModuleList design, that uses a list comprehension that adds the layers
as needed. nn.ModuleList takes care of the bookkeeping on the involved
nn.Parameters and makes the class easily accessible. All of which allow the
ResNet class to be a nn.Sequential composite of ResidualBlocks, which
facilitates the entire class design. Both classes adhere to the general format
of nn.Modules.

### BlackBoxPipe

This class is a both a naive tracer device and provides the training & testing
prodedural pipeline for all models that are inquired by the Bayesian
optimization. Crucially, this class takes the parametrized model (since the
architecture is not searched over), sets up tracking and provides a method "
evaluate_model_with_SGD", which can be passed as function object to the
Bayesian Optimizer (BO) - but is still aware and part of the BlackBoxPipe
tracer. This function configures the pipeline in dependence to the
hyperparameter that is to be optimized; in this case: the SGD's learning rate
parameter. It's output is the cost, which BO inquired for a given proposed
hyperparameter.

### Bayesian Optimisation

Since the Optimisation Problem at hand is 1d, the current implementation is
confined to such spaces. Target is to find the lambda, that minimizes the loss
function. It does so, by inquiring the first lambda's cost by a randomly chosen
lambda on the provided 1d search space (Top-left image). 

![alt text](./Plots/bo_example-1.png "Title")

With this observation, a Gaussian
Process (here: using ELBO for optimization) can be utilized to approximate the
cost function. Using the GP and its assumption on the
(spatial) correlation structure, both a mean prediction for the function can be
estimated conditional on the observed points as well as a quantification of the
uncertainty is now possible. Predicting mean and variance for each lambda in
the search space, the expected improvement over the current best performing
lambda can be calculated; thereby weighing the functions estimated mean and
variance in an explore & exploit manner. The next candidate obtained in 
this fashion is inquired from the provided closure and the cycle until the 
budget on function evaluations is depleedted.

#### Implementational details

This implementation seeks to optimize only 1d search spaces.

The expected improvement (EI) can be a multimodal function with flat regions
connecting the modes. In turn, this function is very difficult to optimize
using classic SGD flavour. Instead, this implementation uses an ungainly and
naive optimization approach of (cheaply) evaluating the EI on a grid and
choosing its arg max.

Be well aware, that the pyro gp implementation apparently can predict zero &
negative variances. To avoid the numerical intricacies of the problem, such
values are explicitly set to 1e-10 for further computation. The EI will be
close to zero in this case since the standard deviation is a factor in EI's
equation.

In order to be capable of debugging the BO, the BO Unittests utilize an
explicit cost function.

## Refactoring Ideas

* Maybe use another third party GP implementation, that allows to fix or
  gradually change the lenghtscale & variance of the GP kernel. The current
  optimizes both during runtime anew for each new datapoint. This may yield 
  drastically inconsistent estimates of these hyperparameters and produce 
  no continuouity on the smoothness of the cost function.  Further, a GP
  which is used in an online fashion; reusing former results would be
  computationally desirable. All of the above boils down to three advantages
  over the current implementation:
    1) Computational efficiency.
    2) A more stable & consistent GP
    3) the smoothness of the GP (kernel reach) can be fixed by the user.

* Find another arg max procedure for finding the current maximum of the
  expected improvement function. The current implementation simply 
  evaluates ei on a tighly spanned grid over the search space and returns 
  its max. Challenges here are multi-modal distributions and flat 
  (zero-gradient) areas which are hard to traverse through.

* Clean up BO's interface; decide where to put the arguments: either at the 
  class.__init__ or prefereably to optimize. The latter can be advantageous 
  when considering continuation, if the last incumbent does not provide 
  sufficient improvement and more budget is granted.

* Allow for a continuation protocol; i.e. kick off, where it left off. Maybe
  use some common interface similar to torch.optim.Optimizer incl. its step 
  method.

* Add logging.

* Extend to multiple dimensions of the searchspace.



