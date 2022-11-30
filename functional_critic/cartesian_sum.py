import numpy as np
import time
import torch
def full_cartesian_sum(rank, act_dim):
    inner_arg_freq = torch.cartesian_prod(*[torch.arange(0, rank, 1) for i in range(act_dim)])
    return inner_arg_freq


def cartesian_sum(sum_max, act_dim):
    ranks_to_add = list(range(0, sum_max + 1))
    if act_dim == 1: # base case.
        return [[r] for r in ranks_to_add]
    cartesian_products_to_return = []
    for cartesian_pair in cartesian_sum(sum_max, act_dim-1):
        sum_cartesian_pair = sum(cartesian_pair)
        for r in ranks_to_add:
            if (sum_cartesian_pair + r) <= sum_max:
                cartesian_products_to_return.append(cartesian_pair + [r])
    return cartesian_products_to_return

if __name__ == "__name__":
    start1 = time.time()
    print("new method", len(cartesian_sum(7, 5)))
    end1 = time.time()

    print("time", end1 - start1)
