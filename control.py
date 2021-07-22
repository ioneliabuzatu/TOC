import time
import duckie.duckie


def main():
    # Simulate Clean Data _ Steady-State Simulation
    sim = duckie.duckie.Ducky(
        number_genes=100,  # Features
        number_gene_types=9,  # Number of cell types
        cells_to_simulate=100,  # 300,  # Number of single cells for which expression is simulated
        noise_params=1,
        decays=0.8,
        sampling_state=15,
        noise_type='dpd'
    )

    # GRN is Interaction graph of the genes
    # MR is Master Regulator
    # production_rate1, ..., productions_rate(#bins)
    # shared_coop_state is for the hill function
    sim.build_graph(input_file_taregts='duckie/steady-state_input_GRN.txt', input_file_regs='duckie/steady-state_input_MRs.txt', shared_coop_state=2)

    start = time.time()
    sim.simulate()
    expr = sim.getExpressions()
    print(expr)
    print(time.time() - start)


if __name__ == '__main__':
    main()
