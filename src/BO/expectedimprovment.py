from math import pi

import torch
import torch.distributions as td


class ExpectedImprovement:
    def eval(self, lamb, eps=0.):
        """
        Calculate the Expected Improvement.

        function definition based on the lecture slides
        https://learn.ki-campus.org/courses/automl-luh2021/items/7rd8zSXREMWYBfbVTXLcci

        :param lamb: torch.Tensor. The values at which the EI is supposed to be
         evaluated at.
        :param eps: float. This parameter allows the user to tip the balance of
        EI towards exploration or exploitation.
        :return: torch.Tensor. The EI evaluated at lamb.
        """
        with torch.no_grad():
            # Inquire the MAP estimate for mu^t(lamb), sd^t(lamb) from GP
            mu_t, _, sd_t = self.gpr_t.predict(lamb)

        # Calculate EI.
        Z = (self.costs[self.inc_idx] - mu_t - eps) / sd_t
        u = (sd_t
             * (Z * td.Normal(0., 1.).cdf(Z)
                + torch.exp(td.Normal(0., 1.).log_prob(Z))
                # Normalizing constant of std. Normal required only for
                # plotting the exact EI:
                + torch.tensor([2. * pi]) ** -1 / 2))

        return u

    def max_ei(self, precision=200, eps=0.):
        """
        Find the maximum of the previously calculated expected improvement.

        Naive optimization of the Expected improvement.
        This function uses a trivial grid evaluation across the search space
        and evaluates self.expected_improvement for each of these grid points.
        lastly, it returns lamb* = argmax_{lamb} u(lamb).

        :param precision: number of evenly spaced EI evaluations on the
        search space.
        :param eps: float. This parameter allows the user to tip the balance of
        EI towards exploration or exploitation.
        :returns the maximum value of the current EI function
        """
        # Evaluate EI on the entire searchspace.
        lamb = torch.linspace(*self.search_space, steps=precision)
        u = ExpectedImprovement.eval(self, lamb, eps=eps)

        # Find and return lamb = argmax_{lamb} u(lamb)
        argmax = u.max(0)[1]
        return lamb[argmax].reshape((1,))


class ExpextedImprov_grad(ExpectedImprovement):
    def max_ei(self, iteration, eps=0, budget=2000):
        """
        Intent: create multiple initializations (e.g. 10 evenly spaced or
        randomly placed on search space & use e.g. ADAM with some no.steps to
        reach the opt.
        """
        # Find initial values.
        initials = torch.zeros(iteration + 2)
        initials[0], initials[-1] = self.search_space
        initials[1:-1] = self.inquired[:iteration]
        e = 0.001  # to stay in bounds at init.
        initials = (initials[:-1] + initials.diff() - e)  # .detach()

        # Find the peaks of EI.
        initials.requires_grad = True
        optimizer = torch.optim.Adam(params=[initials], lr=0.01)
        for s in range(budget):
            optimizer.zero_grad()

            u = -ExpectedImprovement.eval(self, initials, eps=eps)
            u.backward(gradient=torch.ones_like(initials))

            # Consider, that this actually requires 2nd order derivative due to
            # mu(lamb) & sd(lamb)
            # FIXME: This part seems not to pass the gradients through
            # mu, _, sd = self.gpr_t.predict(initials)
            # mu.backward(gradient=torch.ones_like(initials))
            optimizer.step()

        # TODO Check if still in search space; else replace with bound.

        # Find and return lamb = argmax_{lamb} u(lamb)
        argmax = initials.max(0)[1]
        return initials[argmax].reshape((1,))
