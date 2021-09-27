from utils import get_bits
from bitarray import bitarray


class ImagePartition:
    def __init__(self, n, bits_per_image, activations_with_idx, cutoff, size_partition):
        self.n = n
        self.bits_per_image = bits_per_image
        self.bit_array = n * bits_per_image * bitarray("0")
        for i, item in enumerate(activations_with_idx):
            activation, image_id = item
            start_bit = image_id * bits_per_image
            end_bit = start_bit + bits_per_image
            if i < cutoff:
                self.bit_array[start_bit:end_bit] = False
            else:
                partition = 1 + (i - cutoff) // size_partition
                bits = get_bits(partition, bits_per_image)
                for j, bit_pos in enumerate(range(start_bit, end_bit)):
                    self.bit_array[bit_pos] = bits[j]

    def get_image_ids_by_partition_id(self, partition_id):
        images = set()
        partition_bits = get_bits(partition_id, self.bits_per_image)
        for image_id in range(self.n):
            start_bit = image_id * self.bits_per_image
            end_bit = start_bit + self.bits_per_image
            image_partition = self.bit_array[start_bit:end_bit].tolist()

            same = True
            for i in range(len(partition_bits)):
                if int(partition_bits[i]) != int(image_partition[i]):
                    same = False
                    break
            if same:
                images.add(image_id)

        return images

    def get_partition_id_by_image_id(self, image_id):
        start_bit = image_id * self.bits_per_image
        end_bit = start_bit + self.bits_per_image
        image_partition = self.bit_array[start_bit:end_bit].tolist()
        res = 0
        for bit in image_partition:
            res = (res << 1) | bit
        return res
