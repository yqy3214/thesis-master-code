import numpy as np
from sklearn.cluster import KMeans
import fix_support_barycenter as bc

def update_position(point_lists, transport, w_list):
    ans = []
    for i in range(len(w_list)):
        tmp = transport[i][:len(point_lists[i]), :-1].sum(axis=0)[None,:]
        tmp[tmp==0] = 1
        ans.append(
            w_list[i] * np.einsum('nd,nk->kd', point_lists[i], transport[i][:len(point_lists[i]), :-1] / tmp) 
        )
    return sum(ans)

def barycenter(point_lists, u_list, w_list, zeta_a, zeta_b, barycenter_support_size, z, barycenter = None, div = 3, method='A'):
    # 最优解
    model = bc.fix_support_barycenter(point_lists, barycenter, u_list, w_list, zeta_a, zeta_b)
    if not barycenter is None:
        barycenter_weight_opt, transport_opt, cost_opt = model.get_barycenter()
    else:
        cost_opt = np.inf
    cost = np.inf
    for i in range(max(2, len(point_lists) // div)):
        real_support_size = barycenter_support_size + int(barycenter_support_size * 3 * z // len(point_lists[0]))
        kmeans_ans = KMeans(min(real_support_size, len(point_lists[i])), n_init=3).fit(point_lists[i])
        tmp_barycenter = kmeans_ans.cluster_centers_ 
        
        tmp_barycenter_weight, tmp_transport, tmp_cost = model.update_barycenter_place(barycenter=tmp_barycenter)
        if tmp_cost < cost:
            barycenter_weight, transport, cost, barycenter = tmp_barycenter_weight, tmp_transport, tmp_cost, tmp_barycenter
    
    # maxIter = 10
    # for i in range(maxIter):
    #     new_barycenter = update_position(point_lists, transport, w_list)
    #     center_shift = np.sum((new_barycenter - barycenter) ** 2)
    #     if center_shift <= 1e-4:
    #         break
    #     barycenter = new_barycenter
    #     barycenter_weight, transport, cost = model.update_barycenter_place(barycenter=barycenter)
    
    return barycenter_weight, transport, cost, cost_opt, barycenter
