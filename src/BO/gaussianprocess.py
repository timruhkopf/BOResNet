import pyro
import pyro.contrib.gp as gp
import torch


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
        print(msg.format(
            self.gpr_t.kernel.variance.item(),
            self.gpr_t.kernel.lengthscale.item(),
            self.gpr_t.noise.item()))

    def predict(self, X):
        with torch.no_grad():
            mean, cov = self.gpr_t(X, full_cov=True, noiseless=False)
            var = cov.diag()

            # CAREFULL: pyros' GP may produce negative & zero variance
            # predictions (~= -9e-7) ! to avoid producing nans in the following
            # calculations, they are set to 1e-10 instead.
            var[var <= 0] = 1e-10
            sd = var.sqrt()
        return mean, cov, sd


class GP_constrained:
    def __init__(self):
        """
        Consider unsing this GP implementation:
        https://github.com/cagrell/gp_constr/blob/master/Example_1a.ipynb.
        This would allow to constrain the GP to predict e.g. only positive
        values for both mean & variance - which is handy if the cost
        function (such as MSE or Crossentropy loss) is known to be strictly
        positive.
        """
        raise NotImplementedError()

    def fit_hyperparam(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()
