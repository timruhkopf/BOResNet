import unittest
from pathlib import Path

import torch

from src.bo.botracker import BoTracker
from src.bo.gaussianprocess import GaussianProcess


class Test_BoTracker(unittest.TestCase):

    def setUp(self) -> None:
        self.botracker = BoTracker(search_space=(0., 1.), budget=2, noise=0.)
        self.botracker.inquired = torch.tensor([0., 0.5, 1.])
        self.botracker.costs = torch.tensor([2., 1., 3.])

        # Add GPs to the model
        gp = GaussianProcess(
            x=self.botracker.inquired,
            y=self.botracker.costs,
            initial_var=0.5,
            initial_length=0.1,
            noise=0.)

        self.gpstate = gp.gpr_t.state_dict()
        self.botracker.gprs.append(gp)

        # Make tmp folder structure:
        # mkdir /gpr_models
        self.path = '/home/tim/PycharmProjects/BOResNet/test/tmp/'
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        # TODO Clear the tmp folder
        pass

    def test_saving_loading(self):
        """Check whether the tracker can be recovered fully from disk."""
        self.botracker.save(self.path)
        newbo = BoTracker.load(self.path)

        # Check basic attributes are recovered.
        names = ['search_space', 'budget', 'noise', ]
        for n in names:
            self.assertEqual(
                self.botracker.__getattribute__(n),
                newbo.__getattribute__(n),
                msg='Basic attributes could not be recovered.')

        tensors = ['inquired', 'costs']
        for t in tensors:
            self.assertTrue(torch.equal(
                self.botracker.__getattribute__(t),
                newbo.__getattribute__(t)))

        # Check GP is recovered.
        for p_old, p_new in zip(self.gpstate.values(),
                                newbo.gprs[0].gpr_t.state_dict().values()):
            self.assertTrue(torch.equal(p_old, p_new))


if __name__ == '__main__':
    unittest.main(exit=False)
