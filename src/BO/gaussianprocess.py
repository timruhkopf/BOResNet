import torch
import pyro
import pyro.contrib.gp as gp


class GaussianProcess:
    def __init__(self, x, y, initial_var, initial_length, noise):
        """

        :param x: tensor.
        :param y: tensor.
        :param initial_var: float
        :param initial_length: float
        :param noise: float
        """
        kernel = gp.kernels.Matern32(
            input_dim=1, variance=torch.tensor(initial_var),
            lengthscale=torch.tensor(initial_var))
        self.gpr_t = gp.models.GPRegression(
            x, y, kernel,
            noise=torch.tensor(noise),
            jitter=1e-5)

        self.optimizer = torch.optim.Adam(self.gpr_t.parameters(), lr=0.005)
        self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

    def fit_hyperparam(self, num_steps):
        pyro.clear_param_store()

        losses = []
        for i in range(num_steps):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.gpr_t.model, self.gpr_t.guide)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        msg = 'GP Parameter\n\tVariance: {}\n\tLengthscale: {}\n\tNoise: {}'
        print(msg.format(*self.estimated_gpr_param[-1]))
