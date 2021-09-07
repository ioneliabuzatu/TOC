import collections
import csv
import sys
import time as clock

import jax.numpy as jnp
import jax.numpy as np
import jax.ops
import networkx as nx
import numpy as onp

import utils
from duckie.genes import gene

np.int = int
np.float = float


class sergio:
    def __init__(self, number_genes, number_bins, number_sc, noise_params, noise_type, decays, dynamics=False, sampling_state=10, tol=1e-3, \
                 window_length=100, dt=0.01, optimize_sampling=False, bifurcation_matrix=None, noise_params_splice=None, noise_type_splice=None, \
                 splice_ratio=4, dt_splice=0.01, migration_rate=None):
        self.nGenes_ = number_genes
        self.nBins_ = number_bins
        self.nSC_ = number_sc
        self.coop_state = None
        self.sampling_state_ = sampling_state
        self.tol_ = tol
        self.winLen_ = window_length
        self.dt_ = dt
        self.optimize_sampling_ = optimize_sampling
        self.level2verts_ = {}
        self.gene_to_len = collections.defaultdict(int)
        self.gID_to_level_and_idx = {}  # This dictionary gives the level and idx in self.level2verts_ of a given gene ID
        self.binDict = {}  # This maps bin ID to list of gene objects in that bin; only used for dynamics simulations
        self.maxLevels_ = 0
        self.init_concs_ = np.zeros((number_genes, number_bins))
        self.noiseType_ = noise_type
        self.dyn_ = dynamics
        self.nConvSteps = np.zeros(number_bins)  # This holds the number of simulated steps till convergence
        if dynamics:
            self.bifurcationMat_ = np.array(bifurcation_matrix)
            self.binOrders_ = []
            self.binDict = {}
            for b in range(self.nBins_):
                self.binDict[b] = np.zeros(self.nGenes_, ).tolist()
        ############
        # This graph stores for each vertex: parameters(interaction
        # parameters for non-master regulators and production rates for master
        # regulators), tragets, regulators and level
        ############
        self.graph_ = {}

        if np.isscalar(noise_params):
            self.noise_level = noise_params
        else:
            raise NotImplementedError

        if np.isscalar(decays) == 1:
            self.decayVector_ = np.repeat(decays, number_genes)
        elif np.shape(decays)[0] == number_genes:
            self.decayVector_ = decays
        else:
            print("Error: expect one decay parameter per gene")
            sys.exit()

        if self.dyn_:
            if (self.bifurcationMat_ == None).any():
                print("Error: Bifurcation Matrix is missing")
                sys.exit()

            if noise_type_splice == None:
                self.noiseTypeSp_ = noise_type
            else:
                self.noiseTypeSp_ = noise_type_splice

            if dt_splice == None:
                self.dtSp_ = np.copy(self.dt_)
            else:
                self.dtSp_ = dt_splice

            if noise_params_splice == None:
                self.noiseParamsVectorSp_ = np.copy(self.noise_level)
            elif np.isscalar(noise_params_splice):
                self.noiseParamsVectorSp_ = np.repeat(noise_params_splice, number_genes)
            elif np.shape(noise_params_splice)[0] == number_genes:
                self.noiseParamsVectorSp_ = noise_params_splice
            else:
                print("Error: expect one splicing noise parameter per gene")
                sys.exit()

            if np.isscalar(splice_ratio):
                self.ratioSp_ = np.repeat(splice_ratio, number_genes)
            elif np.shape(splice_ratio)[0] == number_genes:
                self.ratioSp_ = splice_ratio
            else:
                print("Error: expect one splicing ratio parameter per gene")
                sys.exit()

        self.global_state = np.full((self.nBins_, self.nGenes_, self.sampling_state_ * self.nSC_), np.nan)
        self.simulation_time = onp.zeros(self.nGenes_, dtype=int)
        self.half_responses = None

    def mean_state(self, gene_group_id=None, bin=None):
        assert isinstance(bin, (np.ndarray, onp.ndarray, type(None)))
        assert isinstance(gene_group_id, (np.ndarray, onp.ndarray, type(None)))
        # np.full((number_genes, number_bins), np.nan)
        gs = self.global_state[:, :, self.scIndices_]
        me = np.mean(gs, -1)

        if bin is None and gene_group_id is None:
            gg_gene_expression = me
        elif bin is not None and gene_group_id is not None:
            gg_gene_expression = me[bin, gene_group_id]
        elif bin is None:
            gg_gene_expression = me[:, gene_group_id]
        elif gene_group_id is None:
            gg_gene_expression = me[bin, :]
        else:
            raise NotImplemented

        # self.meanExpression = jax.ops.index_update(self.meanExpression, jax.ops.index[REMOVEME_gene_group_id, :], mean_expressions)
        return gg_gene_expression.T

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
        self.rates = onp.full((self.nGenes_, self.nBins_,), np.nan)
        self.connection_strength = onp.full((self.nGenes_, self.nGenes_), np.nan)
        self.params_shared_coop_state = onp.full((self.nGenes_, self.nGenes_), np.nan)
        self.half_responses = onp.full((self.nGenes_, self.nGenes_), np.nan)
        self._targets = collections.defaultdict(list)
        self._targets_reverse = collections.defaultdict(list)

        with open(input_file_taregts, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            if shared_coop_state <= 0:
                raise NotImplemented("Not tested yet")
                for row in reader:
                    nRegs = int(row[1])
                    ##################### Raise Error ##########################
                    if nRegs == 0:
                        print("Error: a master regulator (#Regs = 0) appeared in input")
                        sys.exit()
                        ############################################################

                    currInteraction = []
                    currParents = []
                    for regId, K, C_state in zip(row[2: 2 + nRegs], row[2 + nRegs: 2 + 2 * nRegs], row[2 + 2 * nRegs: 2 + 3 * nRegs]):
                        currInteraction.append((np.int(regId), np.float(K), np.float(C_state), 0))  # last zero shows half-response, it is modified in another method
                        allRegs.append(np.int(regId))
                        currParents.append(np.int(regId))
                        self.graph_[np.int(regId)]['targets'].append(np.int(row[0]))

                    self.graph_[np.int(row[0])]['params'] = currInteraction
                    self.graph_[np.int(row[0])]['regs'] = currParents
                    self.graph_[np.int(row[0])]['level'] = -1  # will be modified later
                    allTargets.append(np.int(row[0]))

                    # if self.dyn_:
                    #    for b in range(self.nBins_):
                    #        binDict[b].append(gene(np.int(row[0]),'T', b))
            else:
                for indRow, row in enumerate(reader):
                    nRegs = int(float(row[1]))

                    if nRegs == 0:
                        raise Exception("Error: a master regulator (#Regs = 0) appeared in input")

                    currParents = []
                    currInteraction = []
                    row0 = int(float(row[0]))

                    for regId, K, in zip(row[2: 2 + nRegs], row[2 + nRegs: 2 + 2 * nRegs]):
                        regId = int(float(regId))
                        # currInteraction.append((regId, float(K), shared_coop_state, onp.nan))  # last zero shows half-response, it is modified in another method

                        self.connection_strength[row0, regId] = float(K)
                        self.half_responses[row0, regId] = 0.

                        allRegs.append(regId)
                        currParents.append(regId)
                        self._targets[regId].append(row0)
                        self._targets_reverse[row0].append(regId)
                        self.graph_[regId]['targets'].append(row0)

                    self._targets_reverse[row0] = np.array(self._targets_reverse[row0])
                    self.graph_[row0]['regs'] = currParents
                    self.graph_[row0]['level'] = -1  # will be modified later
                    allTargets.append(row0)

                    # if self.dyn_:
                    #    for b in range(self.nBins_):
                    #        binDict[b].append(gene(np.int(row[0]),'T', b))

        # self.master_regulators_idx_ = set(np.setdiff1d(allRegs, allTargets))
        with open(input_file_regs, 'r') as f:
            masterRegs = []
            for row in csv.reader(f, delimiter=','):
                row0 = int(float(row[0]))
                if np.shape(row)[0] != self.nBins_ + 1:
                    raise Exception("Error: Inconsistent number of bins")

                masterRegs.append(row0)
                self.rates[row0] = [float(i) for i in row[1:]]
                self.graph_[row0]['regs'] = []
                self.graph_[row0]['level'] = -1

                # if self.dyn_:
                #    for b in range(self.nBins_):
                #        binDict[b].append(gene(np.int(row[0]),'MR', b))

        self.master_regulators_idx_ = set(masterRegs)

        if len(self.master_regulators_idx_) + np.shape(allTargets)[0] != self.nGenes_:
            raise Exception("Error: Inconsistent number of genes")

        self.find_levels_()  # make sure that this modifies the graph

        if self.dyn_:
            self.find_bin_order_(self.bifurcationMat_)

        self.rates = jnp.array(self.rates)
        self._targets = {k: sorted(v) for k, v in self._targets.items()}
        self.half_responses = jnp.full((self.nGenes_, self.nGenes_), np.nan)

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
        self.level2verts_[currLayer] = []
        idx = 0

        while U != V:
            # targets = self.targets[v]
            currVerts = set(filter(lambda v: set(self.graph_[v]['targets']).issubset(Z), V - U))

            for v in currVerts:
                self.graph_[v]['level'] = currLayer
                U.add(v)
                if {v}.issubset(self.master_regulators_idx_):
                    allBinList = [gene(self, v, 'MR', i) for i in range(self.nBins_)]
                    self.level2verts_[currLayer].append(allBinList)
                    self.gID_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1
                else:
                    allBinList = [gene(self, v, 'T', i) for i in range(self.nBins_)]
                    self.level2verts_[currLayer].append(allBinList)
                    self.gID_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1

            currLayer += 1
            Z = Z.union(U)
            self.level2verts_[currLayer] = []
            idx = 0

        self.level2verts_.pop(currLayer)
        self.maxLevels_ = currLayer - 1

        if not self.dyn_:
            self.set_scIndices_()

    def set_scIndices_(self, safety_steps=0):
        """
        # First updates sampling_state_ if optimize_sampling_ is set True: to optimize run time,
        run for less than 30,000 steps in first level
        # Set the single cell indices that are sampled from steady-state steps
        # Note that sampling should be performed from the end of Concentration list
        # Note that this method should run after building graph(and layering) and should
        be run just once!
        """

        if self.optimize_sampling_:
            state = np.true_divide(30000 - safety_steps * self.maxLevels_, self.nSC_)
            if state < self.sampling_state_:
                self.sampling_state_ = state

        indices = onp.random.randint(low=-self.sampling_state_ * self.nSC_, high=0, size=self.nSC_)
        self.scIndices_ = self.global_state.shape[-1] + indices

    def calculate_required_steps_(self, level, safety_steps=0):
        """
        # Calculates the number of required simulation steps after convergence at each level.
        # safety_steps: estimated number of steps required to reach convergence (same), although it is not neede!
        """
        # TODO: remove this safety step

        return self.sampling_state_ * self.nSC_ + level * safety_steps

    def calculate_half_response_(self, level):
        """
        Calculates the half response for all interactions between previous layer
        and current layer
        """

        currGenes = self.level2verts_[level]

        for g in currGenes:  # g is list of all bins for a single gene
            gg_idx = g[0].ID
            if not g[0].is_master_regulator:
                # for target in self.targets[gg_idx]:
                regIdxes, = np.where(~np.isnan(self.connection_strength[gg_idx]))
                meanArr = self.mean_state(regIdxes)
                assert np.all(~np.isnan(meanArr))

                self.half_responses = jax.ops.index_update(self.half_responses, jax.ops.index[gg_idx, regIdxes], np.mean(meanArr, axis=1))
            # Else: g is a master regulator and does not need half response

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

    def hill_(self, reg_conc, half_response, repressive):
        # print("use batch")
        num_incoming_edges, num_bins = reg_conc.shape  # or is it outgoing?
        assert half_response.shape == repressive.shape == (num_incoming_edges,)
        num = np.power(reg_conc.T, self.coop_state)
        denom = np.power(half_response.T, self.coop_state) + num

        response = np.true_divide(num, denom)
        is_active = reg_conc.astype(bool)
        response = jax.ops.index_update(response, jax.ops.index[:, repressive], 1 - response[:, repressive])
        return response + repressive * (~is_active).T

    def init_gene_bin_conc_(self, level):
        """
        Initilizes the concentration of all genes in the input level

        Note: calculate_half_response_ should be run before this method
        """

        currGenes = self.level2verts_[level]

        for g in currGenes:
            gg_idx = g[0].ID
            times = self.simulation_time[gg_idx]
            assert times == 0

            if g[0].is_master_regulator:
                allBinRates = self.rates[gg_idx]  # TODO: make batch

                x0 = np.true_divide(np.array(allBinRates), self.decayVector_[gg_idx])

                x0 = np.clip(x0, 0)
                self.global_state = jax.ops.index_update(self.global_state, jax.ops.index[:, gg_idx, times], x0)

            else:
                gene_ids = self._targets_reverse[gg_idx]
                half_responses = self.half_responses[gg_idx, gene_ids]
                connection_strength = self.connection_strength[gg_idx, gene_ids]
                # for interTuple in params:
                #     gene_id, magnitude, coop_state, half_response = interTuple
                repressives = connection_strength < 0
                current_expressions = self.mean_state(gene_ids)
                new_state_concentration = self.hill_(current_expressions, half_responses, repressives)
                rates = np.abs(connection_strength) * new_state_concentration
                gg_rate = rates.sum(1)
                decay = self.decayVector_[g[0].ID]

                x0 = np.true_divide(gg_rate, decay)
                x0 = np.clip(x0, 0)
                assert not np.any(np.isnan(x0))
                self.global_state = jax.ops.index_update(self.global_state, jax.ops.index[:, gg_idx, times], x0)

        # gg_ids = onp.array([g[0].ID for g in currGenes])
        # is_master_regulator = onp.array([g[0].is_master_regulator for g in currGenes])

        # valid_shape = len(gg_ids), self.nBins_

        # mr_global_idx = [idx for (idx, mr) in zip(gg_ids, is_master_regulator) if mr]
        # not_mr_global_idx = [idx for (idx, mr) in zip(gg_ids, is_master_regulator) if not mr]
        # mr_input_idx = np.where(is_master_regulator)
        # not_mr_input_idx, = np.where(~is_master_regulator)
        # _, num_bins = self.rates.shape
        # allBinRates = self.rates[gg_ids, :]

        # if not_mr_global_idx:
        #     half_responses = self.half_responses[not_mr_global_idx, :]  # ng
        #     connection_strength = self.connection_strength[not_mr_global_idx, :]  # ng
        #     # coop_states = self.params_shared_coop_state[not_mr_global_idx, :]  # ng
        #     current_state = self.mean_state()  # gb
        #     assert self.right_expr(current_state, not_mr_global_idx)

        #     # All of them are at the same time so we don't use times = self.simulation_time[not_mr_global_idx]
        #     # current_expressions = np.expand_dims(current_state, 0).repeat(len(not_mr_input_idx), 0)

        #     repressives = connection_strength < 0

        #     new_state_concentration = self.hill_batch(current_state, half_responses, repressives)  # oib
        #     # gg_rate = rates.sum(1)  # b

        #     # rates = np.einsum("oi,oib->oib", np.abs(connection_strength), new_state_concentration)  # 3, 2x3 -> 2x3 , bg
        #     connection_strength = np.expand_dims(connection_strength, -1).repeat(self.nBins_, -1)
        #     k = np.multiply(np.abs(connection_strength), new_state_concentration)  # TODO: is this element wise multiply?
        #     k_active = jax.ops.index_update(k, np.isnan(k), 0.)
        #     k_total = k_active.sum(1)
        #     allBinRates = jax.ops.index_update(allBinRates, jax.ops.index[mr_input_idx], k_total)

        # decay = self.decayVector_[gg_ids]
        # decay = decay.reshape(*decay.shape, 1).repeat(num_bins, -1)
        # x0 = np.true_divide(allBinRates, decay)
        # x0 = np.clip(x0, 0)

        # assert not np.any(np.isnan(x0))
        # assert np.all(x0 == self.global_state[:, gg_ids, times].T)

    def calculate_prod_rate_fast(self, is_master_regulator, target_genes):
        valid_shape = len(target_genes), self.nBins_
        return self.rates[target_genes]

        mr_global_idx = [idx for (idx, mr) in zip(target_genes, is_master_regulator) if mr]
        not_mr_global_idx = [idx for (idx, mr) in zip(target_genes, is_master_regulator) if not mr]
        mr_input_idx = np.where(is_master_regulator)
        not_mr_input_idx = np.where(~is_master_regulator)

        if np.all(is_master_regulator):
            return self.rates[mr_global_idx, :]

        prod_rate = np.zeros(valid_shape)
        prod_rate = jax.ops.index_update(prod_rate, jax.ops.index[mr_input_idx, :], self.rates[mr_global_idx, :])
        del mr_global_idx
        del mr_input_idx

        half_responses = self.half_responses[not_mr_global_idx, :]

        assert self.right_nans(half_responses, not_mr_global_idx)
        # targets = np.array(regIndices)

        connection_strength = self.connection_strength[not_mr_global_idx, :]
        assert self.right_nans(connection_strength, not_mr_global_idx)

        currStep = self.simulation_time[not_mr_global_idx]

        assert np.all(currStep == currStep[0])  # simplification, everything is at the same time anyway
        # TODO: why does this assert fail?!
        curr_step = currStep[0]

        past_concentrations = self.global_state[:, :, curr_step]
        assert self.right_nans(past_concentrations, not_mr_global_idx, weak=True)

        repressives = connection_strength < 0

        new_state_concentration = self.hill_batch(past_concentrations.T, half_responses, repressives, self.coop_state)
        assert self.right_nans(new_state_concentration.mean(-1), not_mr_global_idx)
        assert self.right_nans(connection_strength, not_mr_global_idx)

        connection_strength = np.expand_dims(connection_strength, -1).repeat(self.num_bins, -1)
        k = np.multiply(np.abs(connection_strength), new_state_concentration)  # TODO: is this element wise multiply?
        k_active = jax.ops.index_update(k, np.isnan(k), 0.)
        k_total = k_active.sum(1)

        prod_rate = jax.ops.index_update(prod_rate, jax.ops.index[not_mr_input_idx, :], k_total)
        return prod_rate

    def right_expr(self, var, not_mr_global_idx):
        for i, idx in enumerate(not_mr_global_idx):
            for j in sorted(self._targets_reverse[idx]):
                var_i = var[j]
                assert np.all(~np.isnan(var_i))
        return True

    def right_nans(self, var, not_mr_global_idx, weak=False):
        all_targets = []
        for i, idx in enumerate(not_mr_global_idx):
            for j in sorted(self._targets_reverse[idx]):
                all_targets.append((i, j))
                var_i = var[i, j]
                assert not np.isnan(var_i)

        not_nans = list(zip(*np.where(~np.isnan(var))))
        if not weak:
            assert all([a == b for a, b in zip(all_targets, not_nans)])
        return True

    def CLE_simulator_(self, level):
        start = clock.time()
        self.calculate_half_response_(level)
        self.init_gene_bin_conc_(level)
        init_time = clock.time() - start
        start = clock.time()

        nReqSteps = self.calculate_required_steps_(level)
        sim_set = onp.copy(self.level2verts_[level]).tolist()

        pdt = np.power(self.dt_, 0.5)

        while sim_set:
            delIndicesGenes = []
            # print(self.simulation_time[0])
            assert self.global_state[~np.isnan(self.global_state)].min() >= 0

            gg_ids = onp.array([gg[0].ID for gg in sim_set])
            gg_types = np.array([gg[0].is_master_regulator for gg in sim_set])

            times = self.simulation_time[gg_ids]

            xt = self.global_state[:, gg_ids, times].T  # The transpose makes me sad
            assert not np.any(np.isnan(xt))

            prod_rates = self.calculate_prod_rate_fast(gg_types, gg_ids)
            decay = np.einsum("g,gb->gb", self.decayVector_[gg_ids], xt)

            noise = self.calc_noise(xt, decay, gg_ids, prod_rates)

            dx = self.dt_ * (prod_rates - decay) + pdt * noise + self.actions[times, :, gg_ids]
            assert not np.any(np.isnan(dx))

            x1 = xt + dx
            x1 = np.clip(x1, 0)
            assert not np.any(np.isnan(x1))
            self.global_state = jax.ops.index_update(self.global_state, jax.ops.index[:, gg_ids, times + 1], x1.T)
            self.simulation_time[gg_ids] += 1

            for REMOVEME_gene_group_sim_set_id, _gene_group in enumerate(sim_set):
                REMOVEME_gene_group_id = gg_ids[REMOVEME_gene_group_sim_set_id]
                time = times[REMOVEME_gene_group_sim_set_id]
                delIndices = []
                if time == nReqSteps:
                    delIndices = list(range(len(_gene_group)))
                    REMOVEME_gIDX = self.gID_to_level_and_idx[REMOVEME_gene_group_id][1]
                    for seq_idx, gi in enumerate(_gene_group):
                        self.level2verts_[level][REMOVEME_gIDX][gi.binID] = _gene_group

                sim_set[REMOVEME_gene_group_sim_set_id] = [i for j, i in enumerate(_gene_group) if j not in delIndices]

                if not sim_set[REMOVEME_gene_group_sim_set_id]:
                    delIndicesGenes.append(REMOVEME_gene_group_sim_set_id)

            sim_set = [i for j, i in enumerate(sim_set) if j not in delIndicesGenes]

        print(f"init time: {init_time}, simulation time: {clock.time() - start}")
        print()

    @jax.jit
    def calc_noise(self, currExp, decay, gID, prod_rate):
        assert decay.shape == prod_rate.shape
        if self.noiseType_ == 'sp':
            raise NotImplemented
            # This notation is inconsistent with our formulation, dw should
            # include dt^0.5 as well, but here we multipy dt^0.5 later
            dw = np.random.normal(size=len(currExp))
            amplitude = np.multiply(self.noiseParamsVector_[gID], np.power(prod_rate, 0.5))
            noise = np.multiply(amplitude, dw)

        elif self.noiseType_ == "spd":
            raise NotImplemented
            dw = np.random.normal(size=len(currExp))
            amplitude = np.multiply(self.noiseParamsVector_[gID], np.power(prod_rate, 0.5) + np.power(decay, 0.5))
            noise = np.multiply(amplitude, dw)

        elif self.noiseType_ == "dpd":
            # TODO Current implementation is wrong, it should take different noise facotrs (noiseParamsVector_) for production and decay
            # Answer to above TODO: not neccessary! 'dpd' is already different than 'spd'
            dw_p = onp.random.normal(size=currExp.shape)  # TODO: use jnp
            dw_d = onp.random.normal(size=currExp.shape)

            amplitude_p = np.einsum(",gb->gb", self.noise_level, np.power(prod_rate, 0.5))
            amplitude_d = np.einsum(",gb->gb", self.noise_level, np.power(decay, 0.5))

            noise = np.multiply(amplitude_p, dw_p) + np.multiply(amplitude_d, dw_d)
        assert not np.any(np.isnan(noise))
        return noise

    def simulate(self, actions):
        self.actions = actions
        for level in range(self.maxLevels_, -1, -1):
            print("Start simulating new level")
            self.CLE_simulator_(level)
            print("Done with current level")

        return self.getExpressions()

    def getExpressions(self):
        return self.global_state[:, :, self.scIndices_]

    """""""""""""""""""""""""""""""""""""""
        "" Here is the functionality we need for dynamics simulations
        """""""""""""""""""""""""""""""""""""""

    def find_bin_order_(self, bifurcation_matrix):
        """
        This functions is simular to find_levels_ but for bifurcation. It uses functionality of networkx
        package. Bifurcation_matrix is assumed to be a DAG.

        #ToDo: Consider re-coding find_levels_ with networkx
        """

        bifGraphNX = nx.DiGraph(bifurcation_matrix)
        try:
            self.binOrders_ = list(nx.topological_sort(bifGraphNX))
        except:
            print("ERROR: Bifurication graph is assumed to be acyclic, but a cyclic graph was passed.")
            sys.exit()

    def dynamics_CLE_simulator_(self, binID):
        # TODO: add population steps to this function instead of using 10 as default, make sure to modify it in populate_with_parentCells_ as well

        converged = False
        sim_set = self.binDict[binID]  # this is a list of gene object that we are simulating
        nc = len(sim_set[0].Conc)  # This is the number of cells that we evolve in each iteration. This is equal to the number of cells that is initially populated from parent bin

        print("binID: " + str(binID))
        print("number of initial cells: " + str(nc))

        resume = True
        while resume:
            for gID, g in enumerate(sim_set):

                prod_rate_U = self.calculate_prod_rate_U_(gID, binID, nc)
                prod_rate_S = self.calculate_prod_rate_S_(gID, binID, nc)
                currU = [self.binDict[binID][gID].Conc[i][-1] for i in range(nc)]
                currU = np.array(currU)

                decay_U = np.copy(prod_rate_S)
                currS = [self.binDict[binID][gID].Conc_S[i][-1] for i in range(nc)]
                currS = np.array(currS)
                decay_S = np.true_divide(self.decayVector_[gID], self.ratioSp_[gID]) * currS

                """
                calculate noise U
                """
                if self.noiseType_ == 'sp':
                    # This notation is inconsistent with our formulation, dw should
                    # include dt^0.5 as well, but here we multipy dt^0.5 later
                    dw = np.random.normal(size=nc)
                    amplitude = np.multiply(self.noise_level, np.power(prod_rate_U, 0.5))
                    noise_U = np.multiply(amplitude, dw)

                elif self.noiseType_ == "spd":
                    dw = np.random.normal(size=nc)
                    amplitude = np.multiply(self.noise_level, np.power(prod_rate_U, 0.5) + np.power(decay_U, 0.5))
                    noise_U = np.multiply(amplitude, dw)


                elif self.noiseType_ == "dpd":
                    # TODO Current implementation is wrong, it should take different noise facotrs (noiseParamsVector_) for production and decay
                    # Answer to above TODO: not neccessary! 'dpd' is already different than 'spd'
                    dw_p = np.random.normal(size=nc)
                    dw_d = np.random.normal(size=nc)

                    amplitude_p = np.multiply(self.noise_level, np.power(prod_rate_U, 0.5))
                    amplitude_d = np.multiply(self.noise_level, np.power(decay_U, 0.5))
                    noise_U = np.multiply(amplitude_p, dw_p) + np.multiply(amplitude_d, dw_d)

                """
                calculate noise S
                """
                if self.noiseTypeSp_ == 'sp':
                    # This notation is inconsistent with our formulation, dw should
                    # include dt^0.5 as well, but here we multipy dt^0.5 later
                    dw = np.random.normal(size=nc)
                    amplitude = np.multiply(self.noiseParamsVectorSp_[gID], np.power(prod_rate_S, 0.5))
                    noise_S = np.multiply(amplitude, dw)

                elif self.noiseTypeSp_ == "spd":
                    dw = np.random.normal(size=nc)
                    amplitude = np.multiply(self.noiseParamsVectorSp_[gID], np.power(prod_rate_S, 0.5) + np.power(decay_S, 0.5))
                    noise_S = np.multiply(amplitude, dw)


                elif self.noiseTypeSp_ == "dpd":
                    # TODO Current implementation is wrong, it should take different noise facotrs (noiseParamsVector_) for production and decay
                    # Answer to above TODO: not neccessary! 'dpd' is already different than 'spd'
                    dw_p = np.random.normal(size=nc)
                    dw_d = np.random.normal(size=nc)

                    amplitude_p = np.multiply(self.noiseParamsVectorSp_[gID], np.power(prod_rate_S, 0.5))
                    amplitude_d = np.multiply(self.noiseParamsVectorSp_[gID], np.power(decay_S, 0.5))
                    noise_S = np.multiply(amplitude_p, dw_p) + np.multiply(amplitude_d, dw_d)

                curr_dU = self.dt_ * (prod_rate_U - decay_U) + np.power(self.dt_, 0.5) * noise_U
                curr_dS = self.dt_ * (prod_rate_S - decay_S) + np.power(self.dt_, 0.5) * noise_S

                for i in range(nc):
                    if currU[i] + curr_dU[i] < 0:
                        g.Conc[i].append(0)
                    else:
                        g.Conc[i].append(currU[i] + curr_dU[i])

                    if currS[i] + curr_dS[i] < 0:
                        g.Conc_S[i].append(0)
                    else:
                        g.Conc_S[i].append(currS[i] + curr_dS[i])
                    # g.append_Conc(currU[i] + curr_dU[i])
                    # g.append_Conc_S(currS[i] + curr_dS[i])

                    if converged:
                        g.incrementStep()

            converged = self.check_convergence_dynamics_(binID, nc)

            if self.nConvSteps[binID] == 0 and converged:
                self.nConvSteps[binID] = len(self.binDict[binID][0].Conc[0])

            if converged:
                resume = self.resume_after_convergence(binID)

    def simulate_dynamics(self):
        self.calculate_ssConc_()
        for bi in self.binOrders_:
            print("Start simulating new cell type")
            self.populate_with_parentCells_(bi)
            self.dynamics_CLE_simulator_(bi)
            print("Done with current cell type")

    def getExpressions_dynamics(self):
        ret = np.zeros((self.nBins_, self.nGenes_, self.nSC_))
        ret_S = np.zeros((self.nBins_, self.nGenes_, self.nSC_))

        for bi in range(self.nBins_):
            nSimSteps = len(self.binDict[bi][0].Conc[0]) * len(self.binDict[bi][0].Conc)
            randCells = np.random.choice(range(nSimSteps), size=self.nSC_, replace=False)
            for gID in range(self.nGenes_):
                allConcU = np.concatenate(self.binDict[bi][gID].Conc, axis=0)
                allConcS = np.concatenate(self.binDict[bi][gID].Conc_S, axis=0)
                ret[bi, gID, :] = np.take(allConcU, randCells)
                ret_S[bi, gID, :] = np.take(allConcS, randCells)

        return ret, ret_S

    """""""""""""""""""""""""""""""""""""""
        "" This part is to add technical noise
        """""""""""""""""""""""""""""""""""""""

    def outlier_effect(self, scData, outlier_prob, mean, scale):
        """
        This function
        """
        out_indicator = np.random.binomial(n=1, p=outlier_prob, size=self.nGenes_)
        outlierGenesIndx = np.where(out_indicator == 1)[0]
        numOutliers = len(outlierGenesIndx)

        #### generate outlier factors ####
        outFactors = np.random.lognormal(mean=mean, sigma=scale, size=numOutliers)
        ##################################

        scData = np.concatenate(scData, axis=1)
        for i, gIndx in enumerate(outlierGenesIndx):
            scData[gIndx, :] = scData[gIndx, :] * outFactors[i]

        return np.split(scData, self.nBins_, axis=1)

    def lib_size_effect(self, scData, mean, scale):
        """
        This functions adjusts the mRNA levels in each cell seperately to mimic
        the library size effect. To adjust mRNA levels, cell-specific factors are sampled
        from a log-normal distribution with given mean and scale.

        scData: the simulated data representing mRNA levels (concentrations);
        np.array (#bins * #genes * #cells)

        mean: mean for log-normal distribution

        var: var for log-normal distribution

        returns libFactors ( np.array(nBin, nCell) )
        returns modified single cell data ( np.array(nBin, nGene, nCell) )
        """

        # TODO make sure that having bins does not intefere with this implementation
        ret_data = []

        libFactors = np.random.lognormal(mean=mean, sigma=scale, size=(self.nBins_, self.nSC_))
        for binExprMatrix, binFactors in zip(scData, libFactors):
            normalizFactors = np.sum(binExprMatrix, axis=0)
            binFactors = np.true_divide(binFactors, normalizFactors)
            binFactors = binFactors.reshape(1, self.nSC_)
            binFactors = np.repeat(binFactors, self.nGenes_, axis=0)

            ret_data.append(np.multiply(binExprMatrix, binFactors))

        return libFactors, np.array(ret_data)

    def dropout_indicator(self, scData, shape=1, percentile=65):
        """
        This is similar to Splat package

        Input:
        scData can be the output of simulator or any refined version of it
        (e.g. with technical noise)

        shape: the shape of the logistic function

        percentile: the mid-point of logistic functions is set to the given percentile
        of the input scData

        returns: np.array containing binary indactors showing dropouts
        """
        scData = np.array(scData)
        scData_log = np.log(np.add(scData, 1))
        log_mid_point = np.percentile(scData_log, percentile)
        prob_ber = np.true_divide(1, 1 + np.exp(-1 * shape * (scData_log - log_mid_point)))

        binary_ind = np.random.binomial(n=1, p=prob_ber)

        return binary_ind

    def convert_to_UMIcounts(self, scData):
        """
        Input: scData can be the output of simulator or any refined version of it
        (e.g. with technical noise)
        """

        return np.random.poisson(scData)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
        "" This part is to add technical noise to dynamics data
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def outlier_effect_dynamics(self, U_scData, S_scData, outlier_prob, mean, scale):
        """
        This function
        """
        out_indicator = np.random.binomial(n=1, p=outlier_prob, size=self.nGenes_)
        outlierGenesIndx = np.where(out_indicator == 1)[0]
        numOutliers = len(outlierGenesIndx)

        #### generate outlier factors ####
        outFactors = np.random.lognormal(mean=mean, sigma=scale, size=numOutliers)
        ##################################

        U = np.concatenate(U_scData, axis=1)
        S = np.concatenate(S_scData, axis=1)
        for i, gIndx in enumerate(outlierGenesIndx):
            U[gIndx, :] = U[gIndx, :] * outFactors[i]
            S[gIndx, :] = S[gIndx, :] * outFactors[i]

        return np.split(U, self.nBins_, axis=1), np.split(S, self.nBins_, axis=1)

    def lib_size_effect_dynamics(self, U_scData, S_scData, mean, scale):
        """
        """

        # TODO make sure that having bins does not intefere with this implementation
        ret_data_U = []
        ret_data_S = []

        libFactors = np.random.lognormal(mean=mean, sigma=scale, size=(self.nBins_, self.nSC_))
        for binExprU, binExprS, binFactors in zip(U_scData, S_scData, libFactors):
            normalizFactors_U = np.sum(binExprU, axis=0)
            normalizFactors_S = np.sum(binExprS, axis=0)
            binFactors = np.true_divide(binFactors, normalizFactors_U + normalizFactors_S)
            binFactors = binFactors.reshape(1, self.nSC_)
            binFactors = np.repeat(binFactors, self.nGenes_, axis=0)

            ret_data_U.append(np.multiply(binExprU, binFactors))
            ret_data_S.append(np.multiply(binExprS, binFactors))

        return libFactors, np.array(ret_data_U), np.array(ret_data_S)

    def dropout_indicator_dynamics(self, U_scData, S_scData, shape=1, percentile=65):
        """
        """
        scData = np.array(U_scData) + np.array(S_scData)
        scData_log = np.log(np.add(scData, 1))
        log_mid_point = np.percentile(scData_log, percentile)
        U_log = np.log(np.add(U_scData, 1))
        S_log = np.log(np.add(S_scData, 1))
        prob_ber_U = np.true_divide(1, 1 + np.exp(-1 * shape * (U_log - log_mid_point)))
        prob_ber_S = np.true_divide(1, 1 + np.exp(-1 * shape * (S_log - log_mid_point)))

        binary_ind_U = np.random.binomial(n=1, p=prob_ber_U)
        binary_ind_S = np.random.binomial(n=1, p=prob_ber_S)

        return binary_ind_U, binary_ind_S

    def convert_to_UMIcounts_dynamics(self, U_scData, S_scData):
        """
        Input: scData can be the output of simulator or any refined version of it
        (e.g. with technical noise)
        """

        return np.random.poisson(U_scData), np.random.poisson(S_scData)
