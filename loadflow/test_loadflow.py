import unittest

import loadflow


class LoadFlowTest(unittest.TestCase):

    def setUp(self):
        self.ps = loadflow.PowerSystem(4)
        self.ps.r[0, 1] = 0.01
        self.ps.r[1, 2] = 0.005
        self.ps.r[2, 3] = 0.01

        self.ps.x[0, 1] = 0.5
        self.ps.x[1, 2] = 0.25
        self.ps.x[2, 3] = 0.5

        self.ps.b[0, 1] = 0.4
        self.ps.b[1, 2] = 0.2
        self.ps.b[2, 3] = 0.4

        self.ps.bc[1] = 0.1
        self.ps.bc[2] = 0.1

        self.ps.P[1] = - 0.6
        self.ps.Q[1] = - 0.3
        self.ps.P[2] = - 0.6
        self.ps.Q[2] = - 0.3
        self.ps.P[3] = 0.6

        self.ps.V[0] = 1.
        self.ps.theta[0] = 0.
        self.ps.V[3] = 1.

    def test_n(self):
        self.assertEqual(self.ps.n, 4)

    def test_V(self):
        self.lf = loadflow.LoadFlow(self.ps)
        self.lf.calculate()
        print(self.lf.V)
