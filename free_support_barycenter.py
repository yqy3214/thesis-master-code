import fix_support_barycenter as bg
import numpy as np
from sklearn.cluster import KMeans
from barycenter import barycenter as bc

def update_position(point_lists, transport, w_list):
    ans = []
    for i in range(len(w_list)):
        tmp = transport[i][:len(point_lists[i]), :-1].sum(axis=0)[None,:]
        tmp[tmp==0] = 1
        ans.append(
            w_list[i] * np.einsum('nd,nk->kd', point_lists[i], transport[i][:len(point_lists[i]), :-1] / tmp) 
        )
    return sum(ans)

def free_support_barycenter(point_lists, u_list, w_list, zeta_a, zeta_b, barycenter_support_size, z, freeSupportIter = 30, tol = 1e-3, barycenter = None, div = 3, epsilon=1, init_b = None, method='A'):
    real_support_size = barycenter_support_size + int(barycenter_support_size * z // len(point_lists[0]))
    model = bg.fix_support_barycenter(point_lists, barycenter, u_list, w_list, zeta_a, zeta_b, epsilon=epsilon)
    if not barycenter is None:
        barycenter_weight_opt, transport_opt, cost_opt = model.get_barycenter()
    else:
        cost_opt = np.inf
    cost = np.inf
    if not init_b is None:
        barycenter_weight, transport, cost = model.update_barycenter_place(barycenter=init_b)
        barycenter = init_b
    else:
        barycenter_weight, transport, cost, _, barycenter = bc(point_lists, u_list, w_list, zeta_a, zeta_b, barycenter_support_size, z, None, div, method)

    # return barycenter_weight, transport, cost, cost_opt, center
    for i in range(freeSupportIter):
        new_barycenter = update_position(point_lists, transport, w_list)
        center_shift = np.sum((new_barycenter - barycenter) ** 2)
        print(i, cost, center_shift)
        if center_shift <= tol:
            break
        barycenter = new_barycenter
        barycenter_weight, transport, cost = model.update_barycenter_place(barycenter=barycenter)
    return barycenter_weight, transport, cost, cost_opt, barycenter


if __name__ == '__main__':
    
    pass
