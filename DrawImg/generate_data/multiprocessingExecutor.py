from pathos.multiprocessing import ProcessingPool as Pool
import time

from functools import partial


# @jit
# def one_person_roi(x, y, z):
#     print(x, y, z, '\n')
#     for i in range(1000000):
#         z = 0
#     return x


class ParallelExecutor:
    def __init__(self, processes=12, display=False):
        self.processes = processes
        self.display = display

    def ParallelExecute(self, function, iterations, other_params=None):
        start = time.time()
        pool = Pool(self.processes)
        if other_params is not None:
            results = pool.map(partial(function, **other_params), iterations)
        else:
            results = pool.map(function, iterations)
        stop = time.time()
        if self.display:
            print(stop - start)
        return results


# if __name__ == "__main__":
#     x = list(range(500))
#     y = 1
#     z = 2
#     parallel_executor = ParallelExecutor(processes=10, display=False)
#     parallel_executor.ParallelExecute(one_person_roi, iterations=x, other_params={"y": y, "z": z})
