import numpy as np
from datetime import datetime
import os
from pathlib import Path
import time
from generate_point_lists import generate_point_lists
from barycenter import barycenter


def generate_data_ex(point_lists_num, point_lists_size, n_features, barycenter_support_size, z, method='A'):
    '''
    point_lists_num 分布数, point_lists_size 每个分布点数, n_features 维度, barycenter_support_size 字面意思, z outliers数量
    '''
    point_lists, center, weight_on_barycenter, center_list = generate_point_lists(point_lists_num, point_lists_size, n_features, barycenter_support_size, z)
    # center_list 是偏移以后的每个分布的真实center
    
    u_list = np.full((point_lists_num, point_lists[0].shape[0]), 1 / point_lists[0].shape[0])
    w_list = np.full(point_lists_num, 1 / point_lists_num)
    zeta_a = z / point_lists[0].shape[0]
    zeta_b = 0.0
    
    barycenter_weight, transport, cost, cost_opt, _ = barycenter(point_lists, u_list, w_list, zeta_a, zeta_b, barycenter_support_size, z, center, method=method)
    
    return cost, cost_opt, (abs((barycenter_weight / sum(barycenter_weight) - weight_on_barycenter / sum(weight_on_barycenter))).sum() if len(barycenter_weight) == len(weight_on_barycenter) else np.inf)



if __name__ == '__main__':
    method = 'A'
    f = open(os.path.join(Path(os.path.realpath(__file__)).parent, 'output_log', datetime.now().strftime("%Y:%m:%d-%H:%M") + f'-generate_data_ex.log'), 'w')
    point_lists_num, point_lists_size, n_features, barycenter_support_size = 10, 20000, 40, 30
    for point_lists_size in [5000, 10000, 20000]:
        for point_lists_num in range(2, 7, 2):
            for n_features in range(10, 41, 10):
                for barycenter_support_size in range(10, 41, 10):
                    for zz in [0.025*i for i in range(0,7)]:
                        for j in range(5):
                            t = time.time()
                            cost, cost_opt, diff = generate_data_ex(point_lists_num, point_lists_size, n_features, barycenter_support_size, int(point_lists_size * zz), method)
                            f.write(f'{(point_lists_num, point_lists_size, n_features, barycenter_support_size, "{:.3f}".format(zz), j)}: {(cost, cost_opt, time.time() - t)},\n')
                            f.flush()
    pass

