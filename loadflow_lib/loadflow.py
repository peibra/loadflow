import numpy as np


class Nodes:
    _n = 0
    _bc = []
    _P = []
    _Q = []
    _V = []
    _theta = []
    p = []
    _p_content = []
    v = []
    _v_content = []

    def _initialize_p(self):
        Nodes.p = []
        Nodes._p_content = []
        for id in range(Nodes._n):
            if Nodes._P[id] is not None:
                Nodes.p.append(Nodes._P[id])
                Nodes._p_content.append(('P', id))
            if Nodes._Q[id] is not None:
                Nodes.p.append(Nodes._Q[id])
                Nodes._p_content.append(('Q', id))

    def _initialize_v(self):
        Nodes.v = []
        Nodes._v_content = []
        for id in range(Nodes._n):
            if Nodes._theta[id] is None:
                Nodes.v.append(0.)
                Nodes._v_content.append(('θ', id))
            if Nodes._V[id] is None:
                Nodes.v.append(1.)
                Nodes._v_content.append(('V', id))

    def __init__(self, bc=0, P=None, Q=None,
                 V=None, theta=None,
                 is_generator=False):
        self.__id = Nodes._n
        Nodes._n += 1
        self.is_load = not is_generator
        self._is_generator = is_generator
        if self._is_generator:
            self.__bci = 0
        else:
            self.__bci = bc
        Nodes._bc.append(0)
        Nodes._bc[self.id] = bc
        Nodes._modified_par = True
        self.__P = P
        self.__Q = Q
        self.__V = V
        self.__theta = theta
        Nodes._P.append(0)
        Nodes._P[self.id] = P
        Nodes._Q.append(0)
        Nodes._Q[self.id] = Q
        Nodes._V.append(0)
        Nodes._V[self.id] = V
        Nodes._theta.append(0)
        Nodes._theta[self.id] = theta
        self._initialize_p()
        self._initialize_v()

    @classmethod
    @property
    def n(cls):
        return Nodes._n

    @property
    def id(self):
        return self.__id

    @property
    def is_generator(self):
        return self._is_generator

    @property
    def bc(self):
        return self.__bci

    @bc.setter
    def bc(self, bc):
        if self._is_generator:
            self.__bci = 0
        else:
            self.__bci = bc
        Nodes._bc[self.__id] = bc
        Nodes._modified_par = True

    @property
    def P(self):
        return self.__P

    @P.setter
    def P(self, P):
        self.__P = P
        Nodes._P[self.id] = P
        self._initialize_p()

    @property
    def Q(self):
        return self.__Q

    @Q.setter
    def Q(self, Q):
        self.__Q = Q
        Nodes._Q[self.id] = Q
        self._initialize_p()

    @property
    def V(self):
        return self.__V

    @V.setter
    def V(self, V):
        self.__V = V
        Nodes._V[self.id] = V
        self._initialize_v()

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, theta):
        self.__theta = theta
        Nodes._theta[self.id] = theta
        self._initialize_v()

    @classmethod
    def can_calculate_load_flow(cls):
        if len(Nodes.p) != len(Nodes.v) or len(Nodes.p) == 2 * Nodes._n:
            return False
        else:
            return True


class Branches(Nodes):
    def __init__(self):
        self.__r = np.full([Nodes._n, Nodes._n], float('inf'))
        self.__x = np.zeros([Nodes._n, Nodes._n])
        self.__b = np.zeros([Nodes._n, Nodes._n])
        self.__n = Nodes._n

    def set_r(self, node, r):
        i, j = node[0].id, node[1].id
        if i != j:
            self.__r[i, j] = r
            j, i = node[0].id, node[1].id
            self.__r[i, j] = r
            Nodes._modified_par = True

    @property
    def r(self):
        return self.__r

    def set_x(self, node, x):
        i, j = node[0].id, node[1].id
        if i != j:
            self.__x[i, j] = x
            j, i = node[0].id, node[1].id
            self.__x[i, j] = x
            Nodes._modified_par = True

    @property
    def x(self):
        return self.__x

    def set_b(self, node, b):
        i, j = node[0].id, node[1].id
        if i != j:
            self.__b[i, j] = b
            j, i = node[0].id, node[1].id
            self.__b[i, j] = b
            Nodes._modified_par = True

    @property
    def b(self):
        return self.__b

    def _Y_diag(self, i, r, x, b, bc):
        return sum([1 / (R + 1j*X) + 1j*B / 2
                    for j, (R, X, B)
                    in enumerate(zip(r[i], x[i], b[i])) if j != i
                    ]) + 1j*bc[i]

    def _Y_nondiag(self, r, x):
        return - 1 / (r + 1j*x)

    @property
    def Y(self):
        if Nodes._modified_par is False:
            return self.__Y
        self.__Y = np.zeros([Nodes._n, Nodes._n], dtype=complex)
        Nodes._modified_par = False
        bc = np.array(super()._bc)
        di = np.diag_indices(self.__n)
        E = np.eye(self.__n, dtype=int)
        self.__Y[di] = [self._Y_diag(i, self.__r, self.__x,
                                     self.__b, bc)
                        for i in range(self.__n)]
        self.__Y += np.where(E == 0,
                             self._Y_nondiag(self.__r, self.__x), 0)
        return self.__Y

    def can_calculate_load_flow(self):
        for row in self.__r:
            for item in row:
                if item != float('inf'):
                    return True
        return False


class LoadFlow():

    def calculate(self, nodes):

        if not (nodes.can_calculate_load_flow() and
                self.branches.can_calculate_load_flow()):
            raise Exception('計算が出来ません')

        if nodes._modified_par:
            self.Y = self.branches.Y

        n = nodes._n
        n_p = len(nodes.v)
        cnt = 0

        Y = self.Y
        G = Y.real
        B = Y.imag

        V = np.array([V if V is not None else 1. for V in nodes._V])
        theta = np.array([theta if theta is not None else 0.
                          for theta in nodes._theta])

        v = np.array(nodes.v).reshape([n_p, 1])
        p = np.array(nodes.p).reshape([n_p, 1])

        fnc_v = np.zeros(n_p).reshape([n_p, 1])

        while np.linalg.norm(p - fnc_v, ord=np.inf) > 0.001:

            fP = [sum([V[j] * (G[i, j]*np.cos(theta[i] - theta[j])
                               + B[i, j]*np.sin(theta[i] - theta[j]))
                       for j in range(n)]) * V[i] for i in range(n)]

            fQ = [sum([V[j] * (G[i, j]*np.sin(theta[i] - theta[j])
                               - B[i, j]*np.cos(theta[i] - theta[j]))
                       for j in range(n)]) * V[i] for i in range(n)]

            # ヤコビ行列の計算

            dfPi_dthetaj = np.zeros([n, n])
            dfQi_dthetaj = np.zeros([n, n])
            dfPi_dVj = np.zeros([n, n])
            dfQi_dVj = np.zeros([n, n])

            # i == j の時
            di = np.diag_indices(n)

            dfPi_dthetaj[di] = [- V[i]**2 * B[i, i] - fQ[i]
                                for i in range(n)]
            dfQi_dthetaj[di] = [- V[i]**2 * G[i, i] + fP[i]
                                for i in range(n)]
            dfPi_dVj[di] = [V[i] * G[i, i] + fP[i]/V[i]
                            for i in range(n)]
            dfQi_dVj[di] = [- V[i] * B[i, i] + fQ[i]/V[i]
                            for i in range(n)]

            # i != j の時
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

            Jacobian = np.matrix([[J_dict[pj[0]][vj[0]][pj[1], vj[1]]
                                  for vj in nodes._v_content]
                                  for pj in nodes._p_content])

            f_dict = {'P': fP, 'Q': fQ}

            fnc_v[:, 0] = [f_dict[pf[0]][pf[1]]
                           for pf in nodes._p_content]

            v += np.dot(np.linalg.inv(Jacobian), p - fnc_v)

            v_dict = {'V': V, 'θ': theta}

            for i, (vv, id) in enumerate(nodes._v_content):
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

            br = self.branches
            I_dash = [
                [-1j * br.b[i, j]/2 * V_dot[i]
                 + (V_dot[i] - V_dot[j]) / (br.r[i, j] + 1j * br.x[i, j])
                 for j in range(n)
                 ]
                for i in range(n)
            ]
            I_dash = np.array(I_dash)

            self.power = np.tile(V_dot.reshape(
                [n, 1]), n) * np.conjugate(I_dash)
        else:
            self.power = None

    def __init__(self, branches):
        self.branches = branches
        self.Y = branches.Y
