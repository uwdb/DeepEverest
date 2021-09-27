import heapq
import math

from ImagePartition import ImagePartition
from utils import is_power_of_two


class ReverseIndex:
    def __init__(self, activations, ratio):
        cutoff = int(len(activations) * ratio)
        self.activations_with_idx = [(x[1], x[0]) for x in enumerate(activations)]
        self.activations_with_idx = heapq.nlargest(cutoff, self.activations_with_idx)
        self.activations_with_idx.reverse()  # from small to large


class ReverseIndexNRA(ReverseIndex):
    def __init__(self, activations, ratio):
        self.max_activation = max(activations)
        self.min_activation = min(activations)
        super(ReverseIndexNRA, self).__init__(activations, ratio)


class ReverseIndexPartitioned:
    def __init__(self, ratio, n_partitions, activations):
        if not is_power_of_two(n_partitions):
            raise ValueError("n_partitions should be a power of two")
        else:
            bits_per_image = int(math.log(n_partitions, 2))

        n = len(activations)
        cutoff = int(n * ratio)
        activations_with_idx = sorted([(x[1], x[0]) for x in enumerate(activations)], reverse=True)
        self.activations_with_idx = activations_with_idx[:cutoff]
        self.activations_with_idx.reverse()
        size_partition = (n - cutoff) // (n_partitions - 1) + 1
        self.partition_of_image = ImagePartition(n, bits_per_image, activations_with_idx, cutoff, size_partition)
