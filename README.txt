As part of my interviews for a PhD position at the AutoML group located in
Hannover at the Leibniz University, I was tasked the following coding challenge

Objective: Find the optimal learning rate of a neural network

Optimizer: Implement Bayesian Optimization (BO) as a global optimization
    approach use a Gaussian Process as a predictive model (you can use any
    existing GP library) use expected improvement (EI) as an acquisition
    function Use an optimization budget of 10 function evaluations if you
    want to watch some intro videos for BO, you can find our MOOC on AutoML
    on the ai-campus.org For the deep neural network, please follow the
    specifications below: A small ResNet, e.g., a ResNet-9 --- should not take
    too long for training, but should achieve a reasonable performance

Optimizer: SGD

Deep learning framework: PyTorch

Dataset: KMNIST (https://github.com/rois-codh/kmnist)

Plotting: Starting with the second iteration of Bayesian Optimization, plot all
    observations, the posterior mean, uncertainty estimate and the acquisition
    function after each iteration See here for an exemplary plots (on a different
    task):
    https://towardsdatascience.com/shallow-understanding-on-bayesian-optimization-324b6c1f7083

Programming language:
    Python
    PEP8 (found here: https://www.python.org/dev/peps/pep-0008/)
    Doc-Strings
