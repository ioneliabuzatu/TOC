import csv
import sys

import jax.numpy as np
import jax.ops
import networkx as nx
import numpy as onp
import tqdm

import duckie.constants as c
import duckie.genes

# TODO these have to be removed to allow jax to calculate the correct gradients
np.int = int
np.float = float
np.random = onp.random
np.copy = onp.copy


class Ducky:
    def __init__(self, number_genes, number_gene_types, cells_to_simulate, noise_params, noise_type, decays, input_file_taregts, input_file_regs, shared_coop_state=0,
                 dynamics=False, sampling_state=10, tol=1e-3, window_length=100, dt=0.01, optimize_sampling=False, bifurcation_matrix=None, noise_params_splice=None,
                 noise_type_splice=None, splice_ratio=4, dt_splice=0.01):
        """
        Noise is a gaussian white noise process with zero mean and finite variance.
        noise_params: The amplitude of noise in CLE. This can be a scalar to use for all genes or an array with the same size as number_genes.
        Tol: p-Value threshold above which convergence is reached
        window_length: length of non-overlapping window (# time-steps) that is used to realize convergence
        noise_params and decays: Could be an array of length number_genes, or single value to use the same value for all genes
        sampling_state (>=1): single cells are sampled from sampling_state * number_sc steady-state steps
        optimize_sampling: useful for very large graphs. If set True, may help finding a more optimal sampling_state and so may ignore the input sampling_state
        noise_type: We consider three types of noise, 'sp': a single intrinsic noise is associated to production process, 'spd': a single intrinsic noise is associated to both
        production and decay processes, 'dpd': two independent intrinsic noises are associated to production and decay processes
        dynamics: whether simulate splicing or not
        bifurcation_matrix: is a numpy array (nBins_ * nBins) of <1 values; bifurcation_matrix[i,j] indicates whether cell type i differentiates to type j or not. Its value
        indicates the rate of transition. If dynamics == True, this matrix should be specified
        noise_params_splice: Same as "noise_params" but for splicing. if not specified, the same noise params as pre-mRNA is used
        noise_type_splice: Same as "noise_type" but for splicing. if not specified, the same noise type as pre-mRNA is used
        splice_ratio: it shows the relative amount of spliced mRNA to pre-mRNA (at steady-state) and therefore tunes the decay rate of spliced mRNA as a function of unspliced mRNA.
         Could be an array of length number_genes, or single value to use the same value for all genes
        dt_splice = time step for integrating splice SDE


        Note1: It's assumed that no two or more bins differentiate into the same new bin i.e. every bin has either 0 or 1 parent bin
        Note2: differentitation rates (e.g. type1 -> type2) specified in bifurcation_matrix specifies the percentage of cells of type2 that are at the vicinity of type1
        """

        self.num_genes = number_genes
        self.num_cell_types = number_gene_types
        self.num_cells = cells_to_simulate
        self.sampling_state_ = sampling_state
        self.tollerance = tol
        self.window_length = window_length
        self.dt = dt
        self.optimize_sampling = optimize_sampling
        self.levels_to_vertices = {}
        self.gene_id_to_level_and_idx = {}  # This dictionary gives the level and idx in self.level2verts_ of a given gene ID
        self.binDict = {}  # This maps bin ID to list of gene objects in that bin; only used for dynamics simulations
        self.max_levels = 0
        self.init_concentrations = np.zeros((number_genes, number_gene_types))
        self.simulation_length = self.sampling_state_ * self.num_cells

        # TODO: This should be a list of self.simulation_length x (self.num_cell_types, self.num_genes)
        self.system_state = np.zeros((self.num_cell_types, self.num_genes, self.simulation_length + 1))  # +1 is due to the extra intial  state

        self.mean_expression = -1 * np.ones((number_genes, number_gene_types))
        self.noise_type = noise_type
        self.dynamics = dynamics
        self.steps_to_convergence = np.zeros(number_gene_types)  # This holds the number of simulated steps till convergence
        if dynamics:
            self.bifurcationMat_ = np.array(bifurcation_matrix)
            self.binOrders_ = []
            self.binDict = {}
            for b in range(self.num_cell_types):
                self.binDict[b] = np.zeros(self.num_genes, ).tolist()
        ############
        # This graph stores for each vertex: parameters(interaction
        # parameters for non-master regulators and production rates for master
        # regulators), tragets, regulators and level
        ############
        self.graph_ = {}

        if np.isscalar(noise_params):
            self.noiseParamsVector_ = np.repeat(noise_params, number_genes)
        elif np.shape(noise_params)[0] == number_genes:
            self.noiseParamsVector_ = noise_params
        else:
            raise Exception("Error: expect one noise parameter per gene")

        if np.isscalar(decays) == 1:
            self.decayVector_ = np.repeat(decays, number_genes)
        elif np.shape(decays)[0] == number_genes:
            self.decayVector_ = decays
        else:
            raise Exception("Error: expect one decay parameter per gene")

        if self.dynamics:
            if (self.bifurcationMat_ is None).any():
                print("Error: Bifurcation Matrix is missing")
                sys.exit()

            if noise_type_splice is None:
                self.noiseTypeSp_ = noise_type
            else:
                self.noiseTypeSp_ = noise_type_splice

            if dt_splice is None:
                self.dtSp_ = np.copy(self.dt)
            else:
                self.dtSp_ = dt_splice

            if noise_params_splice is None:
                self.noiseParamsVectorSp_ = np.copy(self.noiseParamsVector_)
            elif np.isscalar(noise_params_splice):
                self.noiseParamsVectorSp_ = np.repeat(noise_params_splice, number_genes)
            elif np.shape(noise_params_splice)[0] == number_genes:
                self.noiseParamsVectorSp_ = noise_params_splice
            else:
                raise Exception("Error: expect one splicing noise parameter per gene")

            if np.isscalar(splice_ratio):
                self.ratioSp_ = np.repeat(splice_ratio, number_genes)
            elif np.shape(splice_ratio)[0] == number_genes:
                self.ratioSp_ = splice_ratio
            else:
                raise Exception("Error: expect one splicing ratio parameter per gene")

        self.build_graph(input_file_taregts, input_file_regs, shared_coop_state)

    def build_graph(self, input_file_taregts, input_file_regs, shared_coop_state):
        """
        # 1- shared_coop_state: if >0 then all interactions are modeled with that
        # coop state, and coop_states in input_file_taregts are ignored. Otherwise,
        # coop states are read from input file. Reasonbale values ~ 1-3
        # 2- input_file_taregts: a csv file, one row per targets. Columns: Target Idx, #regulators,
        # regIdx1,...,regIdx(#regs), K1,...,K(#regs), coop_state1,...,
        # coop_state(#regs)
        # 3- input_file_regs: a csv file, one row per master regulators. Columns: Master regulator Idx,
        # production_rate1,...,productions_rate(#bins)
        # 4- input_file_taregts should not contain any line for master regulators
        # 5- For now, assume that nodes in graph are either master regulator or
        # target. In other words, there should not be any node with no incomming
        # or outgoing edge! OTHERWISE IT CAUSES ERROR IN CODE.
        # 6- The indexing of genes start from 0. Also, the indexing used in
        # input files should match the indexing (if applicable) used for initilizing
        # the object.
        """

        for i in range(self.num_genes):
            self.graph_[i] = {}
            self.graph_[i]['targets'] = []

        allRegs = []
        allTargets = []

        with open(input_file_taregts, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            if shared_coop_state <= 0:
                for row in reader:
                    nRegs = np.int(row[1])
                    if nRegs == 0:
                        raise Exception("Error: a master regulator (#Regs = 0) appeared in input")

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

            else:
                for indRow, row in enumerate(reader):
                    nRegs = int(float(row[1]))
                    ##################### Raise Error ##########################
                    if nRegs == 0:
                        print("Error: a master regulator (#Regs = 0) appeared in input")
                        sys.exit()
                        ############################################################

                    currInteraction = []
                    currParents = []
                    for regId, K, in zip(row[2: 2 + nRegs], row[2 + nRegs: 2 + 2 * nRegs]):
                        currInteraction.append((np.int(np.float(regId)), np.float(K), shared_coop_state, 0))  # last zero shows half-response, it is modified in another method
                        allRegs.append(np.int(np.float(regId)))
                        currParents.append(np.int(np.float(regId)))
                        self.graph_[np.int(np.float(regId))]['targets'].append(np.int(np.float(row[0])))

                    self.graph_[np.int(np.float(row[0]))]['params'] = currInteraction
                    self.graph_[np.int(np.float(row[0]))]['regs'] = currParents
                    self.graph_[np.int(np.float(row[0]))]['level'] = -1  # will be modified later
                    allTargets.append(np.int(np.float(row[0])))

        with open(input_file_regs, 'r') as f:
            masterRegs = []
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if np.shape(row)[0] != self.num_cell_types + 1:
                    raise Exception("Error: Inconsistent number of bins")

                masterRegs.append(int(float(row[0])))
                self.graph_[int(float(row[0]))]['rates'] = onp.asarray(row[1:], dtype=np.float)
                self.graph_[int(float(row[0]))]['regs'] = []
                self.graph_[int(float(row[0]))]['level'] = -1

        self.master_regulators_idx_ = set(masterRegs)

        if len(self.master_regulators_idx_) + np.shape(allTargets)[0] != self.num_genes:
            raise Exception("Error: Inconsistent number of genes")

        self.find_levels_(self.graph_)  # make sure that this modifies the graph

        if self.dynamics:
            self.find_bin_order_(self.bifurcationMat_)

    def find_levels_(self, graph):
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
        V = set(graph.keys())

        currLayer = 0
        self.levels_to_vertices[currLayer] = []
        idx = 0

        while U != V:
            currVerts = set(filter(lambda v: set(graph[v]['targets']).issubset(Z), V - U))

            for v in currVerts:
                graph[v]['level'] = currLayer
                U.add(v)
                if {v}.issubset(self.master_regulators_idx_):
                    allBinList = [duckie.genes.Gene(v, 'MR', i) for i in range(self.num_cell_types)]
                    self.levels_to_vertices[currLayer].append(allBinList)
                    self.gene_id_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1
                else:
                    allBinList = [duckie.genes.Gene(v, 'T', i) for i in range(self.num_cell_types)]
                    self.levels_to_vertices[currLayer].append(allBinList)
                    self.gene_id_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1

            currLayer += 1
            Z = Z.union(U)
            self.levels_to_vertices[currLayer] = []
            idx = 0

        self.levels_to_vertices.pop(currLayer)
        self.max_levels = currLayer - 1

        if not self.dynamics:
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

        if self.optimize_sampling:
            state = np.true_divide(30000 - safety_steps * self.max_levels, self.num_cells)
            if state < self.sampling_state_:
                self.sampling_state_ = state

        self.scIndices_ = np.random.randint(low=- self.sampling_state_ * self.num_cells, high=0, size=self.num_cells)

    def calculate_half_response_(self, level):
        """ Calculates the half response for all interactions between previous layer and current layer """
        current_genes = self.levels_to_vertices[level]

        for gene_group in current_genes:
            gene_group_id = gene_group[0].ID
            c = 0
            if not gene_group[0].is_master_regulator:
                for interTuple in self.graph_[gene_group_id]['params']:
                    regIdx = interTuple[0]
                    mean_expression = self.mean_expression[regIdx]

                    if np.all(mean_expression == -1):
                        raise Exception("Error: Something's wrong in either layering or simulation. Expression of one or more genes in previous layer was not modeled.")

                    self.graph_[gene_group_id]['params'][c] = (*self.graph_[gene_group_id]['params'][c][:3], np.mean(mean_expression))
                    c += 1
            # Else: gene is a master regulator and does not need half response

    def hill(self, reg_conc, params):
        """
        So far, hill function was used in the code to model 1 interaction at a time.
        So the inputs are single values instead of list or array. Also, it models repression based on this assumption.
        if decided to make it work on multiple interaction, repression should be taken care as well.

        """
        assert params.shape[1:] == (4,)
        assert reg_conc.shape[1:] == (9,)
        _, contribution, coop_state, half_response = params.T
        repressive = contribution < 0

        num = np.power(reg_conc.T, coop_state)
        denom = np.power(half_response.T, coop_state) + num
        response = np.true_divide(num, denom)

        is_active = reg_conc.astype(bool)
        response = jax.ops.index_update(response, jax.ops.index[:, repressive], 1 - response[:, repressive])
        # return response * is_active + repressive * (~is_active)
        return response + repressive * (~is_active).T

    def hill_(self, reg_conc, half_response, coop_state, repressive=False):
        """
        So far, hill function was used in the code to model 1 interaction at a time.
        So the inputs are single values instead of list or array. Also, it models repression based on this assumption.
        if decided to make it work on multiple interaction, repression should be taken care as well.
        """
        num = np.power(reg_conc, coop_state)
        denom = np.power(half_response, coop_state) + num

        response = np.true_divide(num, denom)
        is_active = reg_conc.astype(np.bool)
        if repressive:
            response = 1 - response
        # return response * is_active + repressive * (~is_active)
        return response + repressive * (~is_active)

    def initialize_gene_concentration(self, level):
        """
        Initilizes the cell_type_concentration of all genes in the input level

        Note: calculate_half_response_ should be run before this method
        """

        genes_in_level = self.levels_to_vertices[level]

        for gene_group in genes_in_level:
            gene_group_id = gene_group[0].ID
            is_master_regulator = gene_group[0].is_master_regulator
            group_decay_rate = self.decayVector_[gene_group_id]

            if is_master_regulator:
                gene_group_rates = self.graph_[gene_group_id]['rates']
                decayed_gene_group_rates = np.true_divide(gene_group_rates, group_decay_rate)

                self.update_state(gene_group_id, 0, decayed_gene_group_rates)

                for cell_type_idx, rate in enumerate(decayed_gene_group_rates):
                    gene_group[cell_type_idx].append_concentration(rate)
                    assert self.system_state[cell_type_idx, gene_group_id, 0] == rate

            else:
                params = self.graph_[gene_group_id]['params']
                params = np.array(params)
                gene_ids = params[:, c.gene_ids].astype(int)
                regulator_concentration = self.mean_expression[gene_ids, :]

                contribution = params[:, c.contribution]
                rates = np.abs(contribution) * self.hill(regulator_concentration, params)
                cell_type_concentration = np.true_divide(rates, group_decay_rate)
                total_cell_type_concentration = cell_type_concentration.sum(1)

                self.update_state(gene_group_id, 0, total_cell_type_concentration)

                for cell_type_idx in range(self.num_cell_types):
                    gene_group[cell_type_idx].append_concentration(total_cell_type_concentration[cell_type_idx])

    def update_state(self, gene_group_id, time, concentration):
        # self.system_state = jax.ops.index_update(self.system_state, jax.ops.index[:, gene_group_id, simulation_time + 1], new_concentration)
        assert 0 <= time <= self.simulation_length
        concentration = np.clip(concentration, a_min=0)
        self.system_state = jax.ops.index_update(self.system_state, jax.ops.index[:, gene_group_id, time], concentration)

    def calculate_prod_rate_(self, gene_group, level):
        """
        calculates production rates for the input list of gene objects in different bins but all associated to a single gene ID
        """
        is_master_regulator = gene_group[0].is_master_regulator

        if is_master_regulator:
            rates = self.graph_[gene_group[0].ID]['rates']
            assert len(set([gb.cell_type for gb in gene_group])), "[rates[gb.type] for gb in gene_group]"
            rate = rates[gene_group[0].cell_type]
            return [rate, ] * len(gene_group)

        else:
            params = self.graph_[gene_group[0].ID]['params']
            # Ks = [np.abs(t[1]) for t in params]
            regIndices = [t[0] for t in params]
            cell_types = [gb.cell_type for gb in gene_group]
            currStep = gene_group[0].simulated_steps
            # lastLayerGenes = np.copy(self.levels_to_vertices[level + 1])
            # x0 = np.zeros((len(regIndices), len(cell_types)))
            x0 = []

            params = np.array(params)
            Ks = np.abs(params[:, 1])

            for tupleIdx, gene_group_id in enumerate(regIndices):
                # x0.append([])
                regGeneLevel, regGeneIdx = self.gene_id_to_level_and_idx[gene_group_id]
                regGene_allBins = self.levels_to_vertices[regGeneLevel][regGeneIdx]

                # x0 = []
                x0 = self.system_state[:, gene_group_id, currStep]
                for colIdx, gene_type in enumerate(cell_types):
                    x0_i = x0[gene_type]
                    assert x0_i == regGene_allBins[gene_type]._concentration_history[currStep]
                    # x0[-1].append(x0_i)
                    # x0.append(x0_i)
                # x0 = np.array(x0)

                for colIdx, gene_type in enumerate(cell_types):
                    assert colIdx == gene_type

            x0 = np.array(x0).reshape(1, -1)
            hillMatrix = self.hill(x0, params)
            return np.matmul(Ks, hillMatrix.T)

    def CLE_simulator_(self, level):
        self.calculate_half_response_(level)
        self.initialize_gene_concentration(level)

        to_simulate = np.copy(self.levels_to_vertices[level]).tolist()
        print("There are " + str(len(to_simulate)) + " genes to simulate in this layer")

        pbar = tqdm.tqdm(total=self.simulation_length - 1)
        pbar.update(1)
        simulation_time = 0

        while to_simulate:
            completed_genes = self.simulation_step(level, to_simulate, simulation_time)
            to_simulate = [i for j, i in enumerate(to_simulate) if j not in completed_genes]
            pbar.update(1)
            simulation_time += 1

    def simulation_step(self, level, to_simulate, simulation_time):
        completed_genes = []
        for gene_idx, gene_group in enumerate(to_simulate):
            self.update_group(completed_genes, gene_group, gene_idx, level, to_simulate, simulation_time)
        return completed_genes

    def update_group(self, completed_genes, gene_group, gene_idx, level, to_simulate, simulation_time):
        gene_group_id = gene_group[0].ID
        gene_group_level, gIDX = self.gene_id_to_level_and_idx[gene_group_id]

        current_expression = self.system_state[:, gene_group_id, simulation_time]
        assert np.all(current_expression == np.array([gb._concentration_history[-1] for gb in gene_group]))

        prod_rate = self.calculate_prod_rate_(gene_group, level)
        prod_rate = np.array(prod_rate)
        # Calculate decay rate
        lambda_ = self.decayVector_[gene_group_id]
        decay = np.multiply(lambda_, current_expression)
        # Calculate noise

        noise = self.calculate_noise(len(current_expression), decay, gene_group_id, prod_rate)

        curr_dx = self.dt * (prod_rate - decay) + np.power(self.dt, 0.5) * noise
        new_concentration = self.system_state[:, gene_group_id, simulation_time] + curr_dx + self.actions[simulation_time, :, gene_group_id]
        self.update_state(gene_group_id, simulation_time + 1, new_concentration)

        delIndices = []

        for b_idx, gene in enumerate(gene_group):
            cell_type = gene.cell_type
            assert b_idx == cell_type
            assert gene._concentration_history[-1] == self.system_state[cell_type, gene_group_id, simulation_time]

            gene.append_concentration(new_concentration[cell_type])
            gene.simulated_steps += 1

            # TODO: for the moment all genes evolve in simultaneously so they can't be off sync
            assert len(gene._concentration_history) <= self.simulation_length
            if len(gene._concentration_history) == self.simulation_length:
                if __debug__:
                    gene.scExpression = self.system_state[cell_type, gene_group_id, self.scIndices_]
                # gene.set_scExpression(self.scIndices_)
                # self.mean_expression[gene_group_id, cell_type] = np.mean(gene.scExpression)
                self.mean_expression = jax.ops.index_update(self.mean_expression, jax.ops.index[gene_group_id, cell_type], np.mean(gene.scExpression))

                self.levels_to_vertices[level][gIDX][cell_type] = gene
                delIndices.append(b_idx)

        to_simulate[gene_idx] = []
        for j, i in enumerate(gene_group):
            if j not in delIndices:
                to_simulate[gene_idx].append(i)
        if not to_simulate[gene_idx]:
            completed_genes.append(gene_idx)

    def calculate_noise(self, size, decay, gene_group_id, prod_rate):
        if self.noise_type == 'sp':
            raise NotImplementedError
            # This notation is inconsistent with our formulation, dw should
            # include dt^0.5 as well, but here we multipy dt^0.5 later
            dw = np.random.normal(size=size)
            amplitude = np.multiply(self.noiseParamsVector_[gene_group_id], np.power(prod_rate, 0.5))
            noise = np.multiply(amplitude, dw)

        elif self.noise_type == "spd":
            raise NotImplementedError
            dw = np.random.normal(size=size)
            amplitude = np.multiply(self.noiseParamsVector_[gene_group_id], np.power(prod_rate, 0.5) + np.power(decay, 0.5))
            noise = np.multiply(amplitude, dw)

        elif self.noise_type == "dpd":
            # TODO Current implementation is wrong, it should take different noise facotrs (noiseParamsVector_) for production and decay
            # Answer to above TODO: not neccessary! 'dpd' is already different than 'spd'
            dw_p = np.random.normal(size=size)
            dw_d = np.random.normal(size=size)

            amplitude_p = np.multiply(self.noiseParamsVector_[gene_group_id], np.power(prod_rate, 0.5))
            amplitude_d = np.multiply(self.noiseParamsVector_[gene_group_id], np.power(decay, 0.5))

            noise = np.multiply(amplitude_p, dw_p) + np.multiply(amplitude_d, dw_d)
        return noise

    def simulate(self, actions):
        self.actions = actions
        for level in range(self.max_levels, -1, -1):
            print("Start simulating new level")
            self.CLE_simulator_(level)
            print("Done with current level")

    def get_last_state(self):
        if __debug__:
            ret = onp.zeros((self.num_cell_types, self.num_genes, self.num_cells))
            for currGeneBins in self.levels_to_vertices.values():
                for gene_group in currGeneBins:
                    group_id = gene_group[0].ID

                    for gb in gene_group:
                        ret[gb.cell_type, group_id, :] = gb.scExpression

        assert np.all(self.system_state[:, :, self.scIndices_] == np.array(ret))
        return self.system_state[:, :, -1]

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

    def calculate_ssConc_(self):
        """
        This function calculates the steady state concentrations of both unspliced and spliced RNA in the given bin (cell type).
        Note that this steady state concentration will be used to initilize U and S concentration of this bin (if it's a master bin) and its children (if any)

        Half responses are also computed here by calling its function.
        """
        for level in range(self.max_levels, -1, -1):
            for binID in range(self.num_cell_types):
                currGenes = self.levels_to_vertices[level]

                for g in currGenes:
                    if g[0].Type == 'MR':
                        currRate = self.graph_[g[0].ID]['rates'][binID]
                        self.binDict[binID][g[0].ID] = duckie.genes.Gene(g[0].ID, 'MR', binID)
                        self.binDict[binID][g[0].ID].set_ss_conc_U(np.true_divide(currRate, self.decayVector_[g[0].ID]))
                        self.binDict[binID][g[0].ID].set_ss_conc_S(self.ratioSp_[g[0].ID] * np.true_divide(currRate, self.decayVector_[g[0].ID]))
                    else:
                        params = self.graph_[g[0].ID]['params']
                        currRate = 0
                        for interTuple in params:
                            meanExp = self.mean_expression[interTuple[0], binID]
                            currRate += np.abs(interTuple[1]) * self.hill_(meanExp, interTuple[3], interTuple[2], interTuple[1] < 0)
                            # if binID == 0 and g[0].ID == 0:
                            # print meanExp
                            # print interTuple[3]
                            # print interTuple[2]
                            # print interTuple[1]
                            # print self.hill_(meanExp, interTuple[3], interTuple[2], interTuple[1] < 0)

                        self.binDict[binID][g[0].ID] = duckie.genes.Gene(g[0].ID, 'T', binID)
                        self.binDict[binID][g[0].ID].set_ss_conc_U(np.true_divide(currRate, self.decayVector_[g[0].ID]))
                        self.binDict[binID][g[0].ID].set_ss_conc_S(self.ratioSp_[g[0].ID] * np.true_divide(currRate, self.decayVector_[g[0].ID]))
                    # NOTE This is our assumption for dynamics simulations --> we estimate mean expression of g in b with steady state concentration of U_g in b
                    self.mean_expression[g[0].ID, binID] = self.binDict[binID][g[0].ID].ss_U_
                    # if binID == 0 and g[0].ID == 0:
                    #    print currRate
                    #    print self.decayVector_[g[0].ID]
            if level > 0:
                self.calculate_half_response_(level - 1)

    def populate_with_parentCells_(self, binID):
        """
        This function populates the concentrations of gene objects in the given bin with their parent concentration.
        It is used to initilize the concentrations. The number of population is determined by the bifurcation rates. For master bins, it is randomly
        chosen from a normal distribution with mean 20 and variance 5

        Note: concentrations are calculated by adding a normal noise to the SS concentration of parents. Normal noise has mean zero
        and variance = 0.1 * parent_SS_concentration
        """
        parentBins = self.bifurcationMat_[:, binID]

        if np.count_nonzero(parentBins) > 1:
            print("ERROR: Every cell type is assumed to be differentiated from no or one other cell type; wrong bifurcation matrix.")
            sys.exit()

        elif np.count_nonzero(parentBins) == 1:
            parentBinID = np.nonzero(parentBins)[0][0]
            nPopulation = int(round(self.bifurcationMat_[parentBinID, binID] * self.num_cells))
            # self.nInitCells_[binID] = nPopulation

            # Bifurcation rates of <1/nSC are set to 1/nSC
            if nPopulation < 1:
                nPopulation = 1
        else:
            parentBinID = binID
            nPopulation = int(max(1, np.random.normal(20, 5)))
            # self.nInitCells_[binID] = nPopulation

        for g in self.binDict[binID]:
            varU = np.true_divide(self.binDict[parentBinID][g.ID].ss_U_, 20)
            varS = np.true_divide(self.binDict[parentBinID][g.ID].ss_S_, 20)

            deltaU = np.random.normal(0, varU, size=nPopulation)
            deltaS = np.random.normal(0, varS, size=nPopulation)

            for i in range(len(deltaU)):
                g.append_concentration([self.binDict[parentBinID][g.ID].ss_U_ + deltaU[i]])
                g.append_Conc_S([self.binDict[parentBinID][g.ID].ss_S_ + deltaS[i]])

    def calculate_prod_rate_U_(self, gID, binID, num_c_to_evolve):
        """
        calculate production rate of U in a bunch of cells (num_c_to_evolve) for a gene in a bin
        Retunrs a list of 1 * num_c_to_evolve prod rates
        """
        type = self.binDict[binID][gID].Type
        if (type == 'MR'):
            rates = self.graph_[gID]['rates']
            return [rates[binID] for i in range(num_c_to_evolve)]

        else:
            params = self.graph_[gID]['params']
            Ks = [np.abs(t[1]) for t in params]
            Ks = np.array(Ks)
            regIndices = [t[0] for t in params]
            hillMatrix = np.zeros((len(regIndices), num_c_to_evolve))

            for tupleIdx, ri in enumerate(regIndices):
                currRegConc = [self.binDict[binID][ri].concentration_history[i][-1] for i in range(num_c_to_evolve)]
                for ci, cConc in enumerate(currRegConc):
                    hillMatrix[tupleIdx, ci] = self.hill_(cConc, params[tupleIdx][3], params[tupleIdx][2], params[tupleIdx][1] < 0)

            return np.matmul(Ks, hillMatrix)

    def calculate_prod_rate_S_(self, gID, binID, num_c_to_evolve):
        U = [self.binDict[binID][gID].concentration_history[i][-1] for i in range(num_c_to_evolve)]
        U = np.array(U)
        return self.decayVector_[gID] * U

    def check_convergence_dynamics_(self, binID, num_init_cells):
        numSteps = len(self.binDict[binID][0].concentration_history[0])
        if numSteps < self.num_cells:
            return False
        else:
            nConverged = 0
            for g in self.binDict[binID]:
                if not g.converged_:
                    currConc = [g.concentration_history[i][-10:] for i in range(num_init_cells)]
                    meanU = np.mean(currConc, axis=1)
                    errU = np.abs(meanU - g.ss_U_)

                    if g.ss_U_ < 1:
                        t = 0.2 * g.ss_U_
                    else:
                        t = 0.1 * g.ss_U_
                    # t = np.sqrt(num_init_cells * g.varConvConc_U_)
                    for e in errU:
                        if e < t:
                            g.setConverged()
                            break

                elif not g.converged_S_:
                    currConc = [g.Conc_S[i][-10:] for i in range(num_init_cells)]
                    meanS = np.mean(currConc, axis=1)
                    errS = np.abs(meanS - g.ss_S_)

                    if g.ss_S_ < 1:
                        t = 0.2 * g.ss_S_
                    else:
                        t = 0.1 * g.ss_S_
                    # t = np.sqrt(num_init_cells * g.varConvConc_S_)
                    for e in errS:
                        if e < t:
                            g.setConverged_S()
                            break


                else:
                    nConverged += 1

            if nConverged == self.num_genes:
                return True
            else:
                return False

    def resume_after_convergence(self, binID):
        if self.binDict[binID][0].simulated_steps < self.sampling_state_ * self.steps_to_convergence[binID]:
            return True
        else:
            return False

    def dynamics_CLE_simulator_(self, binID):
        # TODO: add population steps to this function instead of using 10 as default, make sure to modify it in populate_with_parentCells_ as well

        converged = False
        sim_set = self.binDict[binID]  # this is a list of gene object that we are simulating
        nc = len(sim_set[0].concentration_history)
        # This is the number of cells that we evolve in each iteration. This is equal to the number of cells that is initially populated from parent bin

        print("binID: " + str(binID))
        print("number of initial cells: " + str(nc))

        resume = True
        while (resume):
            for gID, g in enumerate(sim_set):

                prod_rate_U = self.calculate_prod_rate_U_(gID, binID, nc)
                prod_rate_S = self.calculate_prod_rate_S_(gID, binID, nc)
                currU = [self.binDict[binID][gID].concentration_history[i][-1] for i in range(nc)]
                currU = np.array(currU)

                decay_U = np.copy(prod_rate_S)
                currS = [self.binDict[binID][gID].Conc_S[i][-1] for i in range(nc)]
                currS = np.array(currS)
                decay_S = np.true_divide(self.decayVector_[gID], self.ratioSp_[gID]) * currS

                """
                calculate noise U
                """
                if self.noise_type == 'sp':
                    # This notation is inconsistent with our formulation, dw should
                    # include dt^0.5 as well, but here we multipy dt^0.5 later
                    dw = np.random.normal(size=nc)
                    amplitude = np.multiply(self.noiseParamsVector_[gID], np.power(prod_rate_U, 0.5))
                    noise_U = np.multiply(amplitude, dw)

                elif self.noise_type == "spd":
                    dw = np.random.normal(size=nc)
                    amplitude = np.multiply(self.noiseParamsVector_[gID], np.power(prod_rate_U, 0.5) + np.power(decay_U, 0.5))
                    noise_U = np.multiply(amplitude, dw)


                elif self.noise_type == "dpd":
                    # TODO Current implementation is wrong, it should take different noise facotrs (noiseParamsVector_) for production and decay
                    # Answer to above TODO: not neccessary! 'dpd' is already different than 'spd'
                    dw_p = np.random.normal(size=nc)
                    dw_d = np.random.normal(size=nc)

                    amplitude_p = np.multiply(self.noiseParamsVector_[gID], np.power(prod_rate_U, 0.5))
                    amplitude_d = np.multiply(self.noiseParamsVector_[gID], np.power(decay_U, 0.5))
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

                curr_dU = self.dt * (prod_rate_U - decay_U) + np.power(self.dt, 0.5) * noise_U
                curr_dS = self.dt * (prod_rate_S - decay_S) + np.power(self.dt, 0.5) * noise_S

                for i in range(nc):
                    if currU[i] + curr_dU[i] < 0:
                        g.concentration_history[i].append(0)
                    else:
                        g.concentration_history[i].append(currU[i] + curr_dU[i])

                    if currS[i] + curr_dS[i] < 0:
                        g.Conc_S[i].append(0)
                    else:
                        g.Conc_S[i].append(currS[i] + curr_dS[i])
                    # g.append_Conc(currU[i] + curr_dU[i])
                    # g.append_Conc_S(currS[i] + curr_dS[i])

                    if converged:
                        g.incrementStep()

            converged = self.check_convergence_dynamics_(binID, nc)

            if self.steps_to_convergence[binID] == 0 and converged:
                self.steps_to_convergence[binID] = len(self.binDict[binID][0].concentration_history[0])

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
        ret = np.zeros((self.num_cell_types, self.num_genes, self.num_cells))
        ret_S = np.zeros((self.num_cell_types, self.num_genes, self.num_cells))

        for bi in range(self.num_cell_types):
            nSimSteps = len(self.binDict[bi][0].concentration_history[0]) * len(self.binDict[bi][0].concentration_history)
            randCells = np.random.choice(range(nSimSteps), size=self.num_cells, replace=False)
            for gID in range(self.num_genes):
                allConcU = np.concatenate(self.binDict[bi][gID].concentration_history, axis=0)
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
        out_indicator = np.random.binomial(n=1, p=outlier_prob, size=self.num_genes)
        outlierGenesIndx = np.where(out_indicator == 1)[0]
        numOutliers = len(outlierGenesIndx)

        #### generate outlier factors ####
        outFactors = np.random.lognormal(mean=mean, sigma=scale, size=numOutliers)
        ##################################

        scData = np.concatenate(scData, axis=1)
        for i, gIndx in enumerate(outlierGenesIndx):
            scData[gIndx, :] = scData[gIndx, :] * outFactors[i]

        return np.split(scData, self.num_cell_types, axis=1)

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

        libFactors = np.random.lognormal(mean=mean, sigma=scale, size=(self.num_cell_types, self.num_cells))
        for binExprMatrix, binFactors in zip(scData, libFactors):
            normalizFactors = np.sum(binExprMatrix, axis=0)
            binFactors = np.true_divide(binFactors, normalizFactors)
            binFactors = binFactors.reshape(1, self.num_cells)
            binFactors = np.repeat(binFactors, self.num_genes, axis=0)

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
        out_indicator = np.random.binomial(n=1, p=outlier_prob, size=self.num_genes)
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

        return np.split(U, self.num_cell_types, axis=1), np.split(S, self.num_cell_types, axis=1)

    def lib_size_effect_dynamics(self, U_scData, S_scData, mean, scale):
        """
        """

        # TODO make sure that having bins does not intefere with this implementation
        ret_data_U = []
        ret_data_S = []

        libFactors = np.random.lognormal(mean=mean, sigma=scale, size=(self.num_cell_types, self.num_cells))
        for binExprU, binExprS, binFactors in zip(U_scData, S_scData, libFactors):
            normalizFactors_U = np.sum(binExprU, axis=0)
            normalizFactors_S = np.sum(binExprS, axis=0)
            binFactors = np.true_divide(binFactors, normalizFactors_U + normalizFactors_S)
            binFactors = binFactors.reshape(1, self.num_cells)
            binFactors = np.repeat(binFactors, self.num_genes, axis=0)

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
