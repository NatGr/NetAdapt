"""computes the fusion of several perf_tables by using mean and gaussian filtering"""

import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter

num_perf_tables = 3
name_perf_tables = 'res-40-2-tf-lite-2-times'

perf_tables = []
for i in range(1, num_perf_tables + 1):
    with open(f'{name_perf_tables}_{i}.pickle', 'rb') as file:
        perf_tables.append(pickle.load(file))

median_perf_tables = {}
for key in perf_tables[0].keys():
    median_perf_tables[key] = gaussian_filter(np.mean(np.array(
        [perf_table[key] for perf_table in perf_tables]), axis=0), sigma=3, mode='nearest')

with open(f'{name_perf_tables}_merge.pickle', 'wb') as file:
    perf_tables.append(pickle.dump(median_perf_tables, file))
