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
        self.conc_len = 0
        self.converged_ = False
        self.converged_S_ = False

    def append_Conc(self, currConc):
        raise NotImplemented
        currConc = np.clip(currConc, 0)
        self.sergio.global_state = jax.ops.index_update(self.sergio.global_state, jax.ops.index[self.binID, self.ID, self.conc_len], currConc)
        self.conc_len += 1

    def set_scExpression(self, list_indices):
        """
        selects input indices from self.Conc and form sc Expression
        """
        self.scExpression = np.array(self.Conc)[list_indices]

    @property
    def Conc(self):
        state_conc = self.sergio.global_state[self.binID, self.ID, :self.conc_len]
        return state_conc
