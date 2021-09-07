import collections
import csv

import jax.numpy as jnp
import jax.ops
import numpy as onp

from duckie.genes import gene

GRN = collections.namedtuple("GRN", "K basal_prod_rate adjacency decay_rate coop_state")


class sergio:
    def __init__(self, number_genes, number_bins, number_sc, noise_params, decays, sampling_state=10, tol=1e-3, window_length=100, dt=0.01):
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

        self.simulation_time = None
        self.half_responses = None
        self.pdt = jnp.power(self.dt_, 0.5)

    def build_graph(self, input_file_taregts, input_file_regs, shared_coop_state=0):
        if shared_coop_state <= 0:
            raise NotImplemented

        self.coop_state = shared_coop_state
        for i in range(self.num_genes):
            self.graph_[i] = {}
            self.graph_[i]['targets'] = []

        allRegs = []
        allTargets = []
        self.basal_prod_rate = onp.zeros((self.num_genes, self.num_bins,))
        self.K = onp.zeros((self.num_genes, self.num_genes))
        self.adjacency = onp.zeros((self.num_genes, self.num_genes))
        self._targets = collections.defaultdict(list)

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
                    self._targets[regId].append(row0)
                    self.graph_[regId]['targets'].append(row0)

                self.graph_[row0]['regs'] = currParents
                self.graph_[row0]['level'] = -1  # will be modified later
                allTargets.append(row0)

        # self.master_regulators_idx_ = set(np.setdiff1d(allRegs, allTargets))
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

        # self.basal_prod_rate = self.basal_prod_rate
        # self.K = self.K
        # self.adjacency = self.adjacency)

        self._targets = {k: sorted(v) for k, v in self._targets.items()}
        self.half_responses = onp.zeros((self.num_genes, self.num_genes))
        self.simulation_time = onp.zeros(self.maxLevels_ + 1, dtype=int)
        global_state = onp.full((1 + self.sampling_state_ * self.num_sc, self.num_genes, self.num_bins), -1, dtype=jnp.float32)
        global_state[0, :, :] = 0.
        self.global_state = onp.array(global_state)
        self.scIndices_ = jnp.array(
            global_state.shape[0] + onp.random.randint(low=-self.sampling_state_ * self.num_sc, high=0, size=self.num_sc)) + 1  # 1 is to avoid selecting the first fake state

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

    def init_gene_bin_conc_(self, grn, gg_ids, time, half_responses):
        x0 = self.global_state[time, :, :]
        b = grn.basal_prod_rate[gg_ids]
        k = grn.K[gg_ids]
        h = half_responses[gg_ids]
        x1_g = self.step(x0, b, k, h, grn.coop_state, grn.decay_rate) / (grn.decay_rate ** 2)  # Decay does not apply at initialization, it's actually twice "undecayed"!
        return x1_g

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
    def step(x0, b, k, h, coop_state, decay_rate):
        p_ij = sergio.transcription_factor_model(coop_state, x0, k, h)
        x1_hat_1 = b + jnp.sum(p_ij, axis=1)  # sum across j, eq 5
        x1_i = jax.nn.relu(x1_hat_1 * decay_rate)
        return x1_i

    @staticmethod
    @jax.jit
    def calculate_prod_rate_fast(grn: GRN, gg_ids, time, global_state, half_responses):
        x0 = global_state[time, :, :]
        # x0 = global_state[time, :, :]
        b = grn.basal_prod_rate[gg_ids]
        k = grn.K[gg_ids]
        h = half_responses[gg_ids]
        x1_g = sergio.step(x0, b, k, h, grn.coop_state, grn.decay_rate)
        # x1 = sergio.step(grn, x0, gg_ids, half_responses)
        # TODO: a sum is missing here

        return x1_g

    def CLE_simulator_(self, level):
        gene_in_level_ids = self.graph_level_ids[level]
        level_time_step = self.simulation_time[level]

        if level == self.maxLevels_:
            self.global_state[level_time_step, gene_in_level_ids, :] = self.grn.basal_prod_rate[gene_in_level_ids] / self.grn.decay_rate
        else:
            self.half_responses = sergio.calculate_half_response_(self.grn, gene_in_level_ids, self.scIndices_, self.global_state, self.half_responses)
            x1_g = self.init_gene_bin_conc_(self.grn, gene_in_level_ids, level_time_step, self.half_responses)
            self.global_state = jax.ops.index_update(self.global_state, jax.ops.index[level_time_step, gene_in_level_ids, :], x1_g)

        while self.simulation_time[level] <= self.nReqSteps:
            time = self.simulation_time[level]
            xt = self.global_state[time, gene_in_level_ids, :]

            prod_rates = self.calculate_prod_rate_fast(self.grn, gene_in_level_ids, level_time_step, self.global_state, self.half_responses)
            assert not jnp.any(jnp.isnan(prod_rates))
            assert jnp.all(prod_rates >= 0)

            noise = self.calc_noise(xt, self.decay_rate, prod_rates)

            dx = self.dt_ * (prod_rates - self.decay_rate) + self.pdt * noise + self.actions[time, :, gene_in_level_ids]
            assert not jnp.any(jnp.isnan(dx))

            x1 = xt + dx
            x1 = jax.nn.relu(x1)

            self.global_state = jax.ops.index_update(self.global_state, jax.ops.index[time + 1, gene_in_level_ids, :], x1)
            self.simulation_time[level] += 1

    def calc_noise(self, xt, decay, prod_rate):
        dw_p = onp.random.normal(size=xt.shape)  # TODO: use jnp
        dw_d = onp.random.normal(size=xt.shape)

        amplitude_p = jnp.einsum(",gb->gb", self.noise_level, jnp.power(prod_rate, 0.5))
        amplitude_d = self.noise_level * jnp.power(decay, 0.5)

        noise = jnp.multiply(amplitude_p, dw_p) + jnp.multiply(amplitude_d, dw_d)
        return noise

    def simulate(self, actions):
        self.actions = actions
        for level in range(self.maxLevels_, -1, -1):
            print("Start simulating new level")
            self.CLE_simulator_(level)
            print("Done with current level")

        return self.global_state[self.scIndices_, :, :]
