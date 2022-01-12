"""Manage power system parameter and calculate load flow.

All you need to know to use this is in the example below. PU method is used
for the units. The number of known P, Q and unknown V, theta should be
equal. If the program fails it will return None for every result.

Example
-------
# -------------------------------------------------------------------------
# |Node 0: V=1., theta=0., (P=unknown), (Q=unknown)|
#         |
#         | r=0.01, x=0.05, b=0.04
#         |
# |Node 1: (V=unknown), (theta=unknown), P=-0.6, Q=-0.3| == |bc=0.1|
#         |
#         | r=0.04, x=0.025, b=0.02
#         |
# |Node 2: (V=unknown), (theta=unknown), P=-0.6, Q=-0.3| == |bc=0.1|
# -------------------------------------------------------------------------

from loadflow import PowerSystem, LoadFlow

ps = PowerSystem(3)  # set up a power system of 3 nodes

ps.r[0, 1] = 0.01  # resistance between node 0 and node 1
ps.r[1, 2] = 0.04

ps.x[0, 1] = 0.05  # reactance
ps.x[1, 2] = 0.025

ps.b[0, 1] = 0.04  # susceptance
ps.b[1, 2] = 0.02

ps.bc[1] = 0.1  # susceptance of node 1 other than the branch connected to it
ps.bc[2] = 0.1

ps.P[1] = - 0.6
ps.Q[1] = - 0.3
ps.P[2] = - 0.6
ps.Q[2] = - 0.3

ps.V[0] = 1.
ps.theta[0] = 0.

lf = LoadFlow(ps)  # set up calculation process
lf.calculate()

print(lf.V)  # prints voltage magnitude
>>> [1.0, 0.9651997474478232, 0.9340151566375318]

print(lf.theta)  # prints voltage phase
>>> [0.0, -0.05906500726151168, -0.06665066281793608]

print(lf.P)  # prints effective power
>>> [[ 0.          1.23602053  0.        ]
     [-1.21841535  0.          0.61841531]
     [ 0.         -0.59999996  0.        ]]

print(lf.Q)  # prints reactive power
>>> [[ 0.          0.50246384  0.        ]
     [-0.37580572 -0.          0.22486348]
     [ 0.         -0.19531394 -0.        ]]

# lf.power returns complex power P + jQ
"""

import copy

import numpy as np


class PowerSystem:
    """Manage power system parameter.

    Parameters
    ----------
    n : int
        The number of nodes in the system.

    Attributes
    ----------
    r : numpy.ndarray(n, n) of float
        `r[i, j]` is the resistance of branch i-j (the default of the
        values is inf).
    x : numpy.ndarray(n, n) of float
        `x[i, j]` is the reactance of branch i-j (the default of the
        values is 0.).
    b : numpy.ndarray(n, n) of float
        `b[i, j]` is the susceptance of branch i-j (the default of the
        values is 0.).
    bc : numpy.ndarray(n,) of float
        `bc[i]` is the susceptance of node[i] (the default of the
        values is 0.).
    Y : numpy.ndarray(n,n) of complex
        `Y` is the node-admittance matrix.

    P : list of float or None
        `P[i]` is the real power of node[i] (the default of the
        values is None).
    Q : list of float or None
        `Q[i]` is the reactive power of node[i] (the default of the
        values is None).
    V : list of float or None
        `V[i]` is the voltage of node[i] (the default of the
        values is None).
    theta : list of float or None
        `Q[i]` is the phase of node[i] (the default of the
        values is None).

    """

    def __init__(self, n):
        self._n = n
        self._r = np.full([n, n], float('inf'))
        self._r_copy = copy.deepcopy(self._r)
        self._x = np.zeros([n, n])
        self._x_copy = copy.deepcopy(self._x)
        self._b = np.zeros([n, n])
        self._b_copy = copy.deepcopy(self._b)
        self.bc = np.zeros(n)
        self.P = [None] * n
        self.Q = [None] * n
        self.V = [None] * n
        self.theta = [None] * n

    def _initialize_vec(self):
        self._p = []
        self._p_content = []
        self._v = []
        self._v_content = []

        for i in range(self._n):
            if self.P[i] is not None:
                self._p.append(self.P[i])
                self._p_content.append(('P', i))
            if self.Q[i] is not None:
                self._p.append(self.Q[i])
                self._p_content.append(('Q', i))
            if self.theta[i] is None:
                self._v.append(0.)
                self._v_content.append(('θ', i))
            if self.V[i] is None:
                self._v.append(1.)
                self._v_content.append(('V', i))

    def _Y_diag(self, i, r, x, b, bc):
        """Calculate the diagonal components of node-admittance matrix Y.

        Parameters
        ----------
        i : int
            Row index.
        r : numpy.ndarray(n, n) of float
            Resistance matrix.
        x : numpy.ndarray(n, n) of float
            Reactance matrix.
        b : numpy.ndarray(n, n) of float
            Admittance matrix.
        bc : numpy.ndarray(n,) of float
            Admittance of node.

        Returns
        -------
        diagonal_component : complex
        """

        return sum([1 / (R + 1j*X) + 1j*B / 2
                    for j, (R, X, B)
                    in enumerate(zip(r[i], x[i], b[i])) if j != i
                    ]) + 1j*bc[i]

    def _Y_nondiag(self, r, x):
        """Calculate the non-diagonal components of admittance matrix Y.

        Parameters
        ----------
        r : float
            Resistance.
        x : float
            Reactance.

        Returns
        -------
        non_diagonal_component : complex
        """

        return - 1 / (r + 1j*x)

    @property
    def n(self):
        """The number of nodes in the system.
        """
        return self._n

    @property
    def r(self):
        """The resistance matrix.

        `r` is accessed just like any other numpy array but automatically
        gets coverted to a symmetric matrix.

        Example
        -------
        >>> some_instance.r
        array([[inf, inf],
               [inf, inf]])

        >>> some_instance.r[1, 0] = 0.1
        >>> some_instance.r
        array([[inf, 0.1],
               [0.1, inf]])
        """
        i, j = np.where(np.not_equal(self._r_copy, self._r))
        self._r[i, j] = self._r_copy[i, j]
        self._r[j, i] = self._r_copy[i, j]
        self._r_copy = copy.deepcopy(self._r)

        return self._r_copy

    @property
    def x(self):
        """The reactance matrix.

        See Also
        --------
        r : The resistance matrix.
        """
        i, j = np.where(np.not_equal(self._x_copy, self._x))
        self._x[i, j] = self._x_copy[i, j]
        self._x[j, i] = self._x_copy[i, j]
        self._x_copy = copy.deepcopy(self._x)

        return self._x_copy

    @property
    def b(self):
        """The susceptance matrix.

        See Also
        --------
        r : The resistance matrix.
        """
        i, j = np.where(np.not_equal(self._b_copy, self._b))
        self._b[i, j] = self._b_copy[i, j]
        self._b[j, i] = self._b_copy[i, j]
        self._b_copy = copy.deepcopy(self._b)

        return self._b_copy

    @property
    def Y(self):
        """The node-admittance matrix calculated from r, x and b.
        """

        self._Y = np.zeros([self._n, self._n], dtype=complex)

        di = np.diag_indices(self._n)
        E = np.eye(self._n, dtype=int)

        self._Y[di] = [self._Y_diag(i, self.r, self.x, self.b, self.bc)
                       for i in range(self._n)]
        self._Y += np.where(E == 0, self._Y_nondiag(self.r, self.x), 0)

        return self._Y


class LoadFlow:
    """Calculate load flow

    Parameters
    ----------
    ps : obj
        Instance of PowerSystem class.

    Attributes
    ----------
    V : list of float
        Results of the voltage magnitude of each node.
    theta : list of float
        Results of the voltage phase of each node.
    power : numpy.ndarray(n,n) of complex
        Results of power flow.
    P : numpy.ndarray(n,n) of float
        Real part of `power` (power.real).
    Q : numpy.ndarray(n,n) of float
        Imaginary part of `power` (power.imag).

    Methods
    -------
    calculate()
        Calculate load flow.
    """

    def __init__(self, ps):
        self.ps = ps

    def calculate(self):
        """Calculate load flow.

        Calculates the load flow of a given power system using
        Newton-Raphson method. The threshold for ending calculation
        is 0.001.
        """
        n = self.ps._n
        cnt = 0

        Y = self.ps.Y
        G = Y.real
        B = Y.imag

        V = np.array([V if V is not None else 1. for V in self.ps.V])
        theta = np.array([theta if theta is not None else 0.
                          for theta in self.ps.theta])

        self.ps._initialize_vec()
        n_p = len(self.ps._p)

        v = np.array(self.ps._v).reshape([n_p, 1])
        p = np.array(self.ps._p).reshape([n_p, 1])

        fnc_v = np.zeros(n_p).reshape([n_p, 1])

        while np.linalg.norm(p - fnc_v, ord=np.inf) > 0.001:

            fP = [sum([V[j] * (G[i, j]*np.cos(theta[i] - theta[j])
                               + B[i, j]*np.sin(theta[i] - theta[j]))
                       for j in range(n)]) * V[i] for i in range(n)]

            fQ = [sum([V[j] * (G[i, j]*np.sin(theta[i] - theta[j])
                               - B[i, j]*np.cos(theta[i] - theta[j]))
                       for j in range(n)]) * V[i] for i in range(n)]

            # Calculate Jacobian

            dfPi_dthetaj = np.zeros([n, n])
            dfQi_dthetaj = np.zeros([n, n])
            dfPi_dVj = np.zeros([n, n])
            dfQi_dVj = np.zeros([n, n])

            # i == j

            di = np.diag_indices(n)

            dfPi_dthetaj[di] = [- V[i]**2 * B[i, i] - fQ[i]
                                for i in range(n)]
            dfQi_dthetaj[di] = [- V[i]**2 * G[i, i] + fP[i]
                                for i in range(n)]
            dfPi_dVj[di] = [V[i] * G[i, i] + fP[i]/V[i]
                            for i in range(n)]
            dfQi_dVj[di] = [- V[i] * B[i, i] + fQ[i]/V[i]
                            for i in range(n)]

            # i != j

            E = np.eye(n, dtype=int)

            ViVj = V.reshape([n, 1]).dot(V.reshape(1, n))
            sin_tmt = np.array([[np.sin(t2 - t1)
                               for t1 in theta] for t2 in theta])
            cos_tmt = np.array([[np.cos(t2 - t1)
                               for t1 in theta] for t2 in theta])

            dfPi_dthetaj += np.where(E == 0,
                                     ViVj * (G * sin_tmt - B * cos_tmt), 0)
            dfQi_dthetaj += np.where(E == 0,
                                     - ViVj * (G * cos_tmt + B * sin_tmt), 0)

            Vi = np.tile(V.reshape([n, 1]), n)
            dfPi_dVj += np.where(E == 0, Vi *
                                 (G * cos_tmt - B * sin_tmt), 0)
            dfQi_dVj += np.where(E == 0, Vi *
                                 (- G * sin_tmt - B * cos_tmt), 0)

            J_dict = {'P': {'θ': dfPi_dthetaj, 'V': dfPi_dVj},
                      'Q': {'θ': dfQi_dthetaj, 'V': dfQi_dVj}}

            Jacobian = np.array([[J_dict[pc[0]][vc[0]][pc[1], vc[1]]
                                  for vc in self.ps._v_content]
                                 for pc in self.ps._p_content])

            f_dict = {'P': fP, 'Q': fQ}

            fnc_v = np.array([f_dict[pc[0]][pc[1]]
                              for pc
                              in self.ps._p_content]).reshape([n_p, 1])

            v += np.dot(np.linalg.inv(Jacobian), p - fnc_v)

            v_dict = {'V': V, 'θ': theta}

            for i, (vv, id) in enumerate(self.ps._v_content):
                v_dict[vv][id] = v[i, 0]

            cnt += 1
            if cnt == 50:
                V = None
                theta = None
                self.V = V
                self.theta = theta
                break

        if V is not None:
            self.V = V.tolist()
            self.theta = theta.tolist()
            V = np.array(self.V).astype('complex')
            theta = np.array(self.theta).astype('complex')

            V_dot = V * [*map(lambda x: np.exp(1j * x), theta)]

            I_prime = [
                [-1j * self.ps.b[i, j]/2 * V_dot[i]
                 + (V_dot[i] - V_dot[j]) /
                    (self.ps.r[i, j] + 1j * self.ps.x[i, j])
                 for j in range(n)
                 ]
                for i in range(n)
            ]
            I_prime = np.array(I_prime)

            self.power = np.tile(V_dot.reshape(
                [n, 1]), n) * np.conjugate(I_prime)
            self.P = self.power.real
            self.Q = self.power.imag
        else:
            self.power = None
            self.P = None
            self.Q = None
