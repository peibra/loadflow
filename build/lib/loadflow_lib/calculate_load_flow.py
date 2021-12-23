import numpy as np


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
