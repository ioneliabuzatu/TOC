import jax.numpy as np
import jax.ops


class gene:

    def __init__(self, sergio, geneID, geneType, binID=-1):

        """
        geneType: 'MR' master regulator or 'T' target
        bindID is optional
        """

        self.ID = geneID
        self.sergio = sergio
        self.Type = geneType
        self.binID = binID
        self._Conc = []
        self.Conc_S = []
        self.dConc = []
        self.k = []  # For dynamics simulation it stores k1 to k4 for Rung-Kutta method, list of size 4 * num_c_to_evolve
        self.k_S = []  # For dynamics simulation it stores k1 to k4 for Rung-Kutta method, list of size 4 * num_c_to_evolve
        self.simulatedSteps_ = 0
        self.conc_len = 0
        self.converged_ = False
        self.converged_S_ = False
        self.ss_U_ = 0  # This is the steady state concentration of Unspliced mRNA
        self.ss_S_ = 0  # This is the steady state concentration of Spliced mRNA
        self.ensure_consistency()

    def append_Conc(self, currConc):
        if isinstance(currConc, list):
            raise Exception
            if currConc[0] < 0:
                self.Conc.append([0])
            else:
                self.Conc.append(currConc)
        else:
            if currConc < 0:
                self._Conc.append(0)
            else:
                self._Conc.append(currConc)

        currConc = np.clip(currConc, 0)
        self.sergio.global_state = jax.ops.index_update(self.sergio.global_state, jax.ops.index[self.binID, self.ID, self.conc_len], currConc)
        self.conc_len += 1
        self.ensure_consistency()

    def append_Conc_S(self, currConc):
        if isinstance(currConc, list):
            if currConc[0] < 0:
                self.Conc_S.append([0])
            else:
                self.Conc_S.append(currConc)
        else:
            if currConc < 0:
                self.Conc_S.append(0)
            else:
                self.Conc_S.append(currConc)
        self.ensure_consistency()

    def incrementStep(self):
        self.ensure_consistency()
        self.simulatedSteps_ += 1

    def set_scExpression(self, list_indices):
        """
        selects input indices from self.Conc and form sc Expression
        """
        self.ensure_consistency()
        self.scExpression = np.array(self.Conc)[list_indices]

    @property
    def Conc(self):
        state_conc = self.sergio.global_state[self.binID, self.ID, :self.conc_len]
        self.ensure_consistency()
        return state_conc

    def ensure_consistency(self):
        state_conc = self.sergio.global_state[self.binID, self.ID, :self.conc_len]
        gene_conc = self._Conc
        for a, b in zip(state_conc, gene_conc):
            assert a == b, (float(a), float(b))
        assert list(state_conc) == gene_conc, (list(state_conc), gene_conc)
