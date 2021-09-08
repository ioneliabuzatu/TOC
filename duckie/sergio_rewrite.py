import collections
import csv

import jax.numpy as jnp
import jax.ops
import numpy as onp

from duckie.genes import gene

GRN = collections.namedtuple("GRN", "K basal_prod_rate adjacency decay_rate coop_state")


class sergio:
    def __init__(self, input_file_taregts, input_file_regs, shared_coop_state=0):
        if shared_coop_state <= 0:
            raise NotImplemented

        number_genes = 100  # Features
        number_bins = 2  # Number of cell types
        number_sc = 2  # 0,  # 300,  # Number of single cells for which expression is simulated
        noise_params = 1
        decays = 0.8
        sampling_state = 3  # 15,
        tol = 1e-3
        window_length = 100
        dt = 0.01

        self.num_genes = number_genes
        self.num_bins = number_bins
        self.num_sc = number_sc
        self.coop_state = None
        self.sampling_state_ = sampling_state
        self.tol_ = tol
        self.winLen_ = window_length
        self.dt_ = dt
        self.graph_level = {}
        self.gene_to_len = collections.defaultdict(int)
        self.gID_to_level_and_idx = {}  # This dictionary gives the level and idx in self.level2verts_ of a given gene ID
        self.maxLevels_ = 0
        self.graph_ = {}
        self.nReqSteps = self.sampling_state_ * self.num_sc

        self.noise_level = noise_params
        self.decay_rate = decays

        self.half_responses = None
        self.pdt = jnp.power(self.dt_, 0.5)
        self.coop_state = shared_coop_state

        for i in range(self.num_genes):
            self.graph_[i] = {}
            self.graph_[i]['targets'] = []

        allRegs = []
        allTargets = []
        self.basal_prod_rate = onp.zeros((self.num_genes, self.num_bins,))
        self.K = onp.zeros((self.num_genes, self.num_genes))
        self.adjacency = onp.zeros((self.num_genes, self.num_genes))

        with open(input_file_taregts, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for indRow, row in enumerate(reader):
                nRegs = int(float(row[1]))

                if nRegs == 0:
                    raise Exception("Error: a master regulator (#Regs = 0) appeared in input")

                currParents = []
                row0 = int(float(row[0]))

                for regId, K, in zip(row[2: 2 + nRegs], row[2 + nRegs: 2 + 2 * nRegs]):
                    regId = int(float(regId))
                    self.K[row0, regId] = float(K)
                    self.adjacency[row0, regId] = 1.

                    allRegs.append(regId)
                    currParents.append(regId)
                    self.graph_[regId]['targets'].append(row0)

                self.graph_[row0]['regs'] = currParents
                self.graph_[row0]['level'] = -1  # will be modified later
                allTargets.append(row0)

        with open(input_file_regs, 'r') as f:
            masterRegs = []
            for row in csv.reader(f, delimiter=','):
                row0 = int(float(row[0]))
                if onp.shape(row)[0] != self.num_bins + 1:
                    raise Exception("Error: Inconsistent number of bins")

                masterRegs.append(row0)
                self.basal_prod_rate[row0] = [float(i) for i in row[1:]]
                self.graph_[row0]['regs'] = []
                self.graph_[row0]['level'] = -1

        self.master_regulators_idx_ = set(masterRegs)
        self.master_regulators_idx = jnp.array(list(set(masterRegs)))

        if len(self.master_regulators_idx_) + onp.shape(allTargets)[0] != self.num_genes:
            raise Exception("Error: Inconsistent number of genes")

        self.find_levels_()  # make sure that this modifies the graph

        self.half_responses = onp.zeros((self.num_genes, self.num_genes))
        self.global_state = global_state = onp.full((self.sampling_state_ * self.num_sc, self.num_genes, self.num_bins), -1, dtype=jnp.float32)

        self.scIndices_ = jnp.array(global_state.shape[0] + onp.random.randint(low=-self.sampling_state_ * self.num_sc, high=0, size=self.num_sc))
        self.grn = GRN(self.K, self.basal_prod_rate, self.adjacency, self.decay_rate, self.coop_state)

    def find_levels_(self):
        """
        # This is a helper function that takes a graph and assigns layer to all
        # verticies. It uses longest path layering algorithm from
        # Hierarchical Graph Drawing by Healy and Nikolovself. A bottom-up
        # approach is implemented to optimize simulator run-time. Layer zero is
        # the last layer for which expression are simulated
        # U: verticies with an assigned layer
        # Z: vertizies assigned to a layer below the current layer
        # V: set of all verticies (genes)

        This also sets a dictionary that maps a level to a matrix (in form of python list)
        of all genes in that level versus all bins
        """
        U = set()
        Z = set()
        V = set(self.graph_.keys())

        currLayer = 0
        self.graph_level[currLayer] = []
        idx = 0

        while U != V:
            # targets = self.targets[v]
            currVerts = set(filter(lambda v: set(self.graph_[v]['targets']).issubset(Z), V - U))

            for v in currVerts:
                self.graph_[v]['level'] = currLayer
                U.add(v)
                if {v}.issubset(self.master_regulators_idx_):
                    allBinList = [gene(self, v, 'MR', i) for i in range(self.num_bins)]
                    self.graph_level[currLayer].append(allBinList)
                    self.gID_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1
                else:
                    allBinList = [gene(self, v, 'T', i) for i in range(self.num_bins)]
                    self.graph_level[currLayer].append(allBinList)
                    self.gID_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1

            currLayer += 1
            Z = Z.union(U)
            self.graph_level[currLayer] = []
            idx = 0

        self.graph_level.pop(currLayer)
        self.graph_level_ids = [None, ] * len(self.graph_level)
        for level, genes_in_level in self.graph_level.items():
            self.graph_level_ids[level] = onp.array([gg[0].ID for gg in genes_in_level])
        self.maxLevels_ = currLayer - 1

    @staticmethod
    @jax.jit
    def calculate_half_response_(grn, gg_ids, scIndices_, global_state, half_responses):
        # In the original implementation MR genes don't have half response, here we calculate it anyway since it will be multiplied by 0 later.
        adjs = grn.adjacency[gg_ids]
        selected_states = global_state[scIndices_, :, :]
        mes = []
        for adj in adjs:  # g is list of all bins for a single gene
            # MR have 0 adj matrix so their HR will be zero.
            valid_gs = jnp.einsum("tgb,g->tgb", selected_states, adj)  # This should be out of the loop
            assert valid_gs.min() >= 0

            mean_expressions = jnp.mean(valid_gs, (0, 2))
            mes.append(mean_expressions)
        mes = jnp.array(mes)
        return jax.ops.index_add(half_responses, jax.ops.index[gg_ids, :], mes)

    @staticmethod
    @jax.jit
    def transcription_factor_model(coop_state, x0, k, half_response):
        num_bins = x0.shape[1]

        k = jnp.expand_dims(k, -1).repeat(num_bins, -1)
        half_response = jnp.expand_dims(half_response, -1).repeat(num_bins, -1)

        eq6_numerator = x0 ** coop_state
        h_i = half_response ** coop_state
        eq6_denominator = h_i + eq6_numerator + 1e-6
        response = k * (eq6_numerator / eq6_denominator)
        repression = (jnp.abs(k) - k) / 2
        return repression - response

    @staticmethod
    @jax.jit
    def step(x0, b, k, h, coop_state):
        p_ij = sergio.transcription_factor_model(coop_state, x0, k, h)
        x1_hat_1 = b + jnp.sum(p_ij, axis=1)  # sum across j, eq 5
        x1_i = jax.nn.relu(x1_hat_1)
        return x1_i

    def simulate_level(self, gg_ids, actions):
        self.half_responses = sergio.calculate_half_response_(self.grn, gg_ids, self.scIndices_, self.global_state, self.half_responses)

        xt = self.step(
            self.global_state[0, :, :],
            self.grn.basal_prod_rate[gg_ids],
            self.grn.K[gg_ids],
            self.half_responses[gg_ids],
            self.grn.coop_state
        ) / self.grn.decay_rate  # Decay does not apply at initialization, it's actually "undecayed"!

        self.global_state = jax.ops.index_update(self.global_state, jax.ops.index[0, gg_ids, :], xt)

        for time in range(self.nReqSteps):
            prod_rates = sergio.step(
                self.global_state[time, :, :],
                self.grn.basal_prod_rate[gg_ids],
                self.grn.K[gg_ids],
                self.half_responses[gg_ids],
                self.grn.coop_state,
            ) * self.grn.decay_rate

            noise = self.calc_noise(xt, self.decay_rate, prod_rates)

            dx = self.dt_ * (prod_rates - self.decay_rate) + self.pdt * noise + actions[time, :]
            x1 = xt + dx
            xt = jax.nn.relu(x1)
            self.global_state = jax.ops.index_update(self.global_state, jax.ops.index[time, gg_ids, :], xt)

    def calc_noise(self, xt, decay, prod_rate):
        dw_p = onp.random.normal(size=xt.shape)  # TODO: use jnp
        dw_d = onp.random.normal(size=xt.shape)

        amplitude_p = jnp.einsum(",gb->gb", self.noise_level, jnp.power(prod_rate, 0.5))
        amplitude_d = self.noise_level * jnp.power(decay, 0.5)

        noise = jnp.multiply(amplitude_p, dw_p) + jnp.multiply(amplitude_d, dw_d)
        return noise

    def simulate(self, actions):
        for level in range(self.maxLevels_, -1, -1):
            gg_ids = self.graph_level_ids[level]
            self.simulate_level(gg_ids, actions[:, gg_ids, :])

        return self.global_state[self.scIndices_, :, :]
