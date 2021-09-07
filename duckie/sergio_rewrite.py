import collections
import csv
import time as clock

import jax.numpy as jnp
import jax.ops
import numpy as onp

from duckie.genes import gene


class sergio:
    def __init__(self, number_genes, number_bins, number_sc, noise_params, decays, sampling_state=10, tol=1e-3, window_length=100, dt=0.01):
        self.nGenes_ = number_genes
        self.nBins_ = number_bins
        self.nSC_ = number_sc
        self.coop_state = None
        self.sampling_state_ = sampling_state
        self.tol_ = tol
        self.winLen_ = window_length
        self.dt_ = dt
        self.graph_level = {}
        self.gene_to_len = collections.defaultdict(int)
        self.gID_to_level_and_idx = {}  # This dictionary gives the level and idx in self.level2verts_ of a given gene ID
        self.binDict = {}  # This maps bin ID to list of gene objects in that bin; only used for dynamics simulations
        self.maxLevels_ = 0
        self.graph_ = {}
        self.nReqSteps = self.sampling_state_ * self.nSC_

        self.noise_level = noise_params
        self.decay_rate = decays

        self.global_state = onp.full((self.sampling_state_ * self.nSC_, self.nGenes_, self.nBins_), -1, dtype=jnp.float32)
        self.simulation_time = None
        self.half_responses = None
        self.pdt = jnp.power(self.dt_, 0.5)

    def mean_state(self, gene_group_id=None, bin=None):
        assert isinstance(bin, (jnp.ndarray, onp.ndarray, type(None)))
        assert isinstance(gene_group_id, (jnp.ndarray, onp.ndarray, type(None)))
        # np.full((number_genes, number_bins), np.nan)
        gs = self.global_state[self.scIndices_, :, :]
        me = jnp.mean(gs, 0)

        if bin is None and gene_group_id is None:
            gg_gene_expression = me
        elif bin is not None and gene_group_id is not None:
            gg_gene_expression = me[gene_group_id, bin]
        elif bin is None:
            gg_gene_expression = me[gene_group_id, :]
        elif gene_group_id is None:
            gg_gene_expression = me[gene_group_id, bin]
        else:
            raise NotImplemented

        return gg_gene_expression

    def build_graph(self, input_file_taregts, input_file_regs, shared_coop_state=0):
        if shared_coop_state <= 0:
            raise NotImplemented

        self.coop_state = shared_coop_state
        for i in range(self.nGenes_):
            self.graph_[i] = {}
            self.graph_[i]['targets'] = []

        allRegs = []
        allTargets = []
        self.params = {}
        self.basal_prod_rate = onp.zeros((self.nGenes_, self.nBins_,))
        self.K = onp.zeros((self.nGenes_, self.nGenes_))
        self.adjacency = onp.zeros((self.nGenes_, self.nGenes_))

        self._targets = collections.defaultdict(list)
        self._targets_reverse = collections.defaultdict(list)

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
                    self._targets_reverse[row0].append(regId)
                    self.graph_[regId]['targets'].append(row0)

                self.graph_[row0]['regs'] = currParents
                self.graph_[row0]['level'] = -1  # will be modified later
                allTargets.append(row0)

        # self.master_regulators_idx_ = set(np.setdiff1d(allRegs, allTargets))
        with open(input_file_regs, 'r') as f:
            masterRegs = []
            for row in csv.reader(f, delimiter=','):
                row0 = int(float(row[0]))
                if onp.shape(row)[0] != self.nBins_ + 1:
                    raise Exception("Error: Inconsistent number of bins")

                masterRegs.append(row0)
                self.basal_prod_rate[row0] = [float(i) for i in row[1:]]
                self.graph_[row0]['regs'] = []
                self.graph_[row0]['level'] = -1

                # if self.dyn_:
                #    for b in range(self.nBins_):
                #        binDict[b].append(gene(np.int(row[0]),'MR', b))

        self.master_regulators_idx_ = set(masterRegs)
        self.master_regulators_idx = jnp.array(list(set(masterRegs)))

        if len(self.master_regulators_idx_) + onp.shape(allTargets)[0] != self.nGenes_:
            raise Exception("Error: Inconsistent number of genes")

        self.find_levels_()  # make sure that this modifies the graph

        self.basal_prod_rate = jnp.array(self.basal_prod_rate)
        self.K = jnp.array(self.K)
        self.adjacency = jnp.array(self.adjacency)

        self._targets = {k: sorted(v) for k, v in self._targets.items()}
        self.half_responses = jnp.zeros((self.nGenes_, self.nGenes_))
        self.simulation_time = onp.zeros(self.maxLevels_ + 1, dtype=int)

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
                    allBinList = [gene(self, v, 'MR', i) for i in range(self.nBins_)]
                    self.graph_level[currLayer].append(allBinList)
                    self.gID_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1
                else:
                    allBinList = [gene(self, v, 'T', i) for i in range(self.nBins_)]
                    self.graph_level[currLayer].append(allBinList)
                    self.gID_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1

            currLayer += 1
            Z = Z.union(U)
            self.graph_level[currLayer] = []
            idx = 0

        self.graph_level.pop(currLayer)
        self.maxLevels_ = currLayer - 1
        self.scIndices_ = jnp.array(self.global_state.shape[0] + onp.random.randint(low=-self.sampling_state_ * self.nSC_, high=0, size=self.nSC_))

    def calculate_half_response_(self, currGenes, gg_ids):
        # In the original implementation MR genes don't have half response, here we calculate it anyway since it will be multiplied by 0 later.
        adjs = self.adjacency[gg_ids]
        gs = self.global_state[self.scIndices_, :, :]
        mes = []
        for g, adj in zip(currGenes, adjs):  # g is list of all bins for a single gene
            # MR have 0 adj matrix so their HR will be zero.
            valid_gs = jnp.einsum("tgb,g->tgb", gs, adj)  # This should be out of the loop
            me = jnp.mean(valid_gs, (0, 2))
            mes.append(me)
        mes = jnp.array(mes)
        self.half_responses = jax.ops.index_add(self.half_responses, jax.ops.index[gg_ids, :], mes)

    @staticmethod
    def hill_batch(state, connection_strength, repressive, coop_state):
        nodes_to_update = connection_strength.shape[0]
        num_nodes, num_bins = state.shape  # or is it nodes_to_update?
        assert connection_strength.shape == repressive.shape == (nodes_to_update, num_nodes)
        connection_strength = np.expand_dims(connection_strength, -1).repeat(num_bins, -1)
        repressive = np.expand_dims(repressive, -1).repeat(num_bins, -1)

        state_sq = np.power(state, coop_state)
        state_sq = np.expand_dims(state_sq, 0).repeat(nodes_to_update, 0)

        strength_sq = np.power(connection_strength, coop_state)
        total = strength_sq + state_sq

        response = np.true_divide(state_sq, total)
        is_active = state.astype(bool)

        idx = np.argwhere(repressive)
        response = jax.ops.index_update(response, jax.ops.index[idx], 1 - response[idx])

        b = repressive * (~is_active)
        return response + b

    def hill_jax(self, k, x, half_response):
        num = jnp.power(x, self.coop_state)
        denom = jnp.power(half_response, self.coop_state) + num + 1e-9
        response = k * (num / denom)
        repression = (jnp.abs(k) - k) / 2
        return repression - response

    def hill_(self, reg_conc, half_response, repressive):
        print("old")
        num_incoming_edges, num_bins = reg_conc.shape  # or is it outgoing?
        assert half_response.shape == repressive.shape == (num_incoming_edges,)
        num = jnp.power(reg_conc.T, self.coop_state)
        denom = jnp.power(half_response.T, self.coop_state) + num

        response = num / denom
        is_active = reg_conc.astype(bool)
        response = jax.ops.index_update(response, jax.ops.index[:, repressive], 1 - response[:, repressive])
        return response + repressive * (~is_active).T

    def init_gene_bin_conc_(self, gg_ids, time):
        x0 = self.global_state[time, :, :]
        x1 = self.step(x0, gg_ids) / (self.decay_rate ** 2)  # Decay does not apply at initialization, it's actually twice "undecayed"!
        x1_g = x1[gg_ids, :]

        self.global_state = jax.ops.index_update(self.global_state, jax.ops.index[time, gg_ids, :], x1_g)

    def eq6or7_onegene(self, gene_i, x0):
        num_bins = self.nBins_
        k = self.K[gene_i]
        half_responses = self.half_responses[gene_i]

        k = jnp.expand_dims(k, -1).repeat(num_bins, -1)
        half_responses = jnp.expand_dims(half_responses, -1).repeat(num_bins, -1)

        eq6_numerator = x0 ** self.coop_state
        h_i = half_responses ** self.coop_state
        eq6_denominator = h_i + eq6_numerator + 1e-6
        response = k * (eq6_numerator / eq6_denominator)
        repression = (jnp.abs(k) - k) / 2
        return repression - response

    @jax.jit
    def step(self, x0, gg_ids):
        assert not jnp.any(jnp.isnan(x0[self.DEBUG_current_genes, :]))

        x1 = jnp.zeros_like(x0)
        x1_hat_1 = self.basal_prod_rate[gg_ids] + jnp.sum(self.eq6or7_onegene(gg_ids, x0), axis=1)  # sum across j, eq 5
        x1_i = jax.nn.relu(x1_hat_1 * self.decay_rate)
        x1 = jax.ops.index_update(x1, jax.ops.index[gg_ids, :], x1_i)
        for gene_i in gg_ids:
            x1_hat = self.basal_prod_rate[gene_i] + jnp.sum(self.eq6or7_onegene(gene_i, x0), axis=0)  # sum across j, eq 5
            assert jnp.all(x1_hat == x1_hat_1[gene_i])
        return x1

    def calculate_prod_rate_fast(self, target_genes, time):
        if time == 0:
            x0 = jnp.zeros_like(self.global_state[time - 1, :, :])
        else:
            x0 = self.global_state[time - 1, :, :]
        x1 = self.step(x0, target_genes)
        # TODO: a sum is missing here

        return x1[target_genes, :]

    def CLE_simulator_(self, level):
        genes_in_level = self.graph_level[level]
        gene_in_level_ids = onp.array([gg[0].ID for gg in genes_in_level])
        self.DEBUG_gg_types = onp.array([gg[0].is_master_regulator for gg in genes_in_level])
        self.DEBUG_current_genes = gene_in_level_ids

        start = clock.time()
        # currGenes = self.graph_level[level]
        self.calculate_half_response_(genes_in_level, gene_in_level_ids)
        hr_time = clock.time() - start
        start = clock.time()
        level_time_step = self.simulation_time[level]
        self.init_gene_bin_conc_(gene_in_level_ids, level_time_step)
        conc_time = clock.time() - start

        start = clock.time()

        for time in range(self.nReqSteps):
            xt = self.global_state[time, gene_in_level_ids, :]

            prod_rates = self.calculate_prod_rate_fast(gene_in_level_ids, level_time_step)
            assert not jnp.any(jnp.isnan(prod_rates))
            assert jnp.all(prod_rates >= 0)

            noise = self.calc_noise(xt, self.decay_rate, prod_rates)

            dx = self.dt_ * (prod_rates - self.decay_rate) + self.pdt * noise + self.actions[time, :, gene_in_level_ids]
            assert not jnp.any(jnp.isnan(dx))

            x1 = xt + dx
            x1 = jax.nn.relu(x1)

            self.global_state = jax.ops.index_update(self.global_state, jax.ops.index[time + 1, gene_in_level_ids, :], x1)
            self.simulation_time[level] += 1

        sim_time = clock.time() - start
        total = conc_time + sim_time + hr_time
        print(f"conc time: {conc_time / total}, half response {hr_time / total}, simulation time: {sim_time / total}")
        print()

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

        return self.global_state[:, :, self.scIndices_]
