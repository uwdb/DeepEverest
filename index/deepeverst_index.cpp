#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <algorithm>
#include <limits>

#include "ThreadPool.h"

extern "C" {

    using namespace std;

    struct indexed_activation
    {
        int index;
        float value;
    };


    inline bool indexed_activation_greater(const indexed_activation &x, const indexed_activation &y)
    {
        return x.value > y.value;
    }


    void do_construct(const int n_partitions, const float ratio, const int bits_per_image, const int n_images,
                        const float* activations, float* sorted_activations, int* idx_sorted_activations, bool* bit_array,
                        float* par_lower_bound, float* par_upper_bound)
    {
        struct indexed_activation *idx_act = (struct indexed_activation *) malloc(n_images * sizeof(indexed_activation));
        for (int i = 0; i < n_images; ++i)
        {
            idx_act[i].index = i;
            idx_act[i].value = activations[i];
        }
        sort(idx_act, idx_act + n_images, &indexed_activation_greater);

        int cutoff = int(n_images * ratio);
        int size_partition = (n_images - cutoff) / (n_partitions - 1) + 1;

        for (int i = 0; i < n_partitions; ++i)
        {
            par_lower_bound[i] = numeric_limits<float>::infinity();
            par_upper_bound[i] = -numeric_limits<float>::infinity();
        }

        for (int i = 0; i < n_images; ++i)
        {
            int image_id = idx_act[i].index;
            int start_bit = image_id * bits_per_image;
            int end_bit = start_bit + bits_per_image;
            int partition;
            if (i < cutoff)
            {
                partition = 0;
                for (int j = start_bit; j < end_bit; ++j)
                    bit_array[j] = 0;
            } else
            {
                partition = 1 + (i - cutoff) / size_partition;
                for (int j = start_bit, k = bits_per_image - 1; j < end_bit; ++j, --k)
                    bit_array[j] = (partition >> k) & 1;
            }
            
            if (idx_act[i].value < par_lower_bound[partition])
                par_lower_bound[partition] = idx_act[i].value;

            if (idx_act[i].value > par_upper_bound[partition])
                par_upper_bound[partition] = idx_act[i].value;
        }

        for (int i = 0; i < cutoff; ++i)
        {
            idx_sorted_activations[i] = idx_act[cutoff - 1 - i].index;
            sorted_activations[i] = idx_act[cutoff - 1 - i].value;
        }
        free(idx_act);
    }


    void do_construct_parallel(const int bits_per_image, const int n_partitions, const float ratio, const int n_neurons, const int n_images,
                                const float** parameters, float** rev_act, int** rev_idx_act, bool** rev_bit_arr, float** par_low_bound, float** par_upp_bound)
    {
        ThreadPool pool(unsigned(n_neurons / 341));
        for (int i = 0; i < n_neurons; ++i)
            pool.enqueue(do_construct, n_partitions, ratio, bits_per_image, n_images,
                            parameters[i], rev_act[i], rev_idx_act[i], rev_bit_arr[i], par_low_bound[i], par_upp_bound[i]);

    }


    void do_construct_serial(const int bits_per_image, const int n_partitions, const float ratio, const int n_neurons, const int n_images,
                                const float** parameters, float** rev_act, int** rev_idx_act, bool** rev_bit_arr, float** par_low_bound, float** par_upp_bound)
    {
        for (int i = 0; i < n_neurons; ++i)
        {
            do_construct(n_partitions, ratio, bits_per_image, n_images,
                            parameters[i], rev_act[i], rev_idx_act[i], rev_bit_arr[i], par_low_bound[i], par_upp_bound[i]);
        }
    }

}