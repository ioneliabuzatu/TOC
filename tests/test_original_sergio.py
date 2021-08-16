import numpy as np
from pathos.multiprocessing import ProcessingPool as pool
from explore_sergio import steady_state
import unittest
from multiprocessing import Pool
import constants_for_tests
from multiprocessing import Process
import sharedmem
import scipy
import duckie.sergio_control


def create_output_for_testing():
    number_genes = 100
    number_bins = 2
    number_sc = 1
    noise_params = 1
    decays = 0.8
    sampling_state = 15
    noise_type = 'dpd'
    input_file_targets = constants_for_tests.input_file_targets
    input_file_regs = constants_for_tests.input_file_regs
    steady_state_expression = steady_state(
        number_genes=number_genes,
        number_bins=number_bins,
        number_sc=number_sc,
        noise_params=noise_params,
        decays=decays,
        sampling_state=sampling_state,
        noise_type=noise_type,
        input_file_targets=input_file_targets,
        input_file_regs=input_file_regs)
    return steady_state_expression


class TestOriginalSergio(unittest.TestCase):
    """" Check if Duckie is good as the original implementation of SERGIO. """

    def test_create(self):
        """ https://stackoverflow.com/questions/3328766/unit-testing-for-stochastic-processes """
        actual_mean, actual_var = self.run_sergio_many_times()
        desired_mean, desired_var = self.run_sergio_many_times()
        self.assertTrue(np.allclose(actual_mean, desired_mean, rtol=1e-05, atol=1e-08))

    def run_sergio_many_times(self, times=50):
        means_actual = np.zeros((times, 100))
        vars_actual = np.zeros((times, 100))
        for seed in range(times):
            steady_state_expression = create_output_for_testing()
            means_actual[seed] = steady_state_expression.mean(0)
            vars_actual[seed] = steady_state_expression.var(0)

        actual_mean, actual_var = means_actual.mean(0), vars_actual.mean(0)
        return actual_mean, actual_var

    def test_single_simulation(self):
        steady_state_expression = create_output_for_testing()
        print(steady_state_expression.mean())
        return steady_state_expression

    def test_get_expression(self):
        means = []
        for i in range(1):
            steady_state_expression_actual = create_output_for_testing()
            means.append(steady_state_expression_actual.mean())
        print(np.array(means).mean())

    def test_get_expression_with_duckie(self):
        number_genes = 100
        number_bins = 2
        number_sc = 1
        noise_params = 1
        decays = 0.8
        sampling_state = 3
        noise_type = 'dpd'
        input_file_targets = constants_for_tests.input_file_targets
        input_file_regs = constants_for_tests.input_file_regs
        shared_coop_state = 2

        duckie_slow_jax = duckie.sergio_control.sergio(
            number_genes=number_genes,
            number_bins=number_bins,
            number_sc=number_sc,
            noise_params=noise_params,
            decays=decays,
            sampling_state=sampling_state,
            noise_type=noise_type,
        )

        duckie_slow_jax.build_graph(input_file_targets, input_file_regs, shared_coop_state)
        duckie_slow_jax.simulate(actions=None)
        expression = duckie_slow_jax.get_expressions()
        print(expression.mean())

    def test_share_memory(self):
        pool = Pool(processes=20)
        results = [pool.apply_async(self.single_simulation()) for _ in range(10)]
        roots = [r.get() for r in results]
        print(roots)

    def test_with_pathos_sim(self):
        pool = Pool(processes=4)
        list_start_vals = range(10)
        array_2D = np.array(pool.map(self.single_simulation(), list_start_vals))
        pool.close()
        print(array_2D)


if __name__ == '__main__':
    unittest.main()
