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
