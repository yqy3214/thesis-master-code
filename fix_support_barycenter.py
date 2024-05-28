import gurobipy
import numpy as np
from ot import dist

class fix_support_barycenter():
    def __init__(self, point_lists, barycenter: np.ndarray, u_list, w_list, zeta_a, zeta_b, epsilon = 1) -> None:
        '''
        u_list: weight of each point
        w_list: weight of each distribution
        zeta_a: outliers in each distribution
        zeta_b: outliers in barycenter
        '''
        self.zeta_a = zeta_a
        self.zeta_b = zeta_b
        self.epsilon = epsilon
        self.point_lists = point_lists
        
        for i in range(len(u_list)):
            u_list[i] /= sum(u_list[i]) * (1 - zeta_a) #放缩，inliers权重为1
        self.u_list = u_list

        if not isinstance(w_list, np.ndarray):
            w_list = np.array(w_list)
        w_list /= w_list.sum()
        self.w_list = w_list
        
        if barycenter is None:
            self.barycenter = barycenter
            return
        
        if not isinstance(barycenter, np.ndarray):
            barycenter = np.array(barycenter)
        self.barycenter = barycenter

        self.max_point_num = max([point_list.shape[0] for point_list in self.point_lists])
        for i in range(len(self.point_lists)):
            if not isinstance(self.point_lists[i], np.ndarray):
                self.point_lists[i] = np.array(self.point_lists[i])
        
        self.robust_cost = np.zeros((len(self.point_lists), self.max_point_num + 1, self.barycenter.shape[0] + 1))
        for i in range(len(self.point_lists)):
            point_list = self.point_lists[i]
            self.robust_cost[i][:len(point_list), :-1] = dist(point_list, self.barycenter) * self.w_list[i]

        self.model = gurobipy.Model()
        self.model.setParam('OutputFlag', 0)
        # add_variables
        self.barycenter_weight_vars = self.model.addVars(range(barycenter.shape[0]), lb=0, ub=1, vtype=gurobipy.GRB.CONTINUOUS, name='barycenter_weight')
        trans_index = [(i, j, k) for i in range(len(w_list)) for j in range(len(point_lists[i]) + 1) for k in range(len(barycenter) + 1)]
        self.transport_vars = self.model.addVars(trans_index, lb=0, ub=1, obj={i:self.robust_cost[i] for i in trans_index}, vtype=gurobipy.GRB.CONTINUOUS, name='transport')
        
        self.max_weight = self.model.addVar(lb=0, ub=1, obj=epsilon, vtype=gurobipy.GRB.CONTINUOUS, name='max_weight')
        self.model.update()

        self.model.addConstrs(self.transport_vars.sum(i, j, '*') == u_list[i][j] for i in range(len(w_list)) for j in range(len(point_lists[i])))
        self.model.addConstrs(self.transport_vars.sum(i, len(point_lists[i]), '*') <= zeta_b / (1 - zeta_b) for i in range(len(w_list))) # 到dummy point
        self.model.addConstrs(self.transport_vars.sum(i, '*', k) == self.barycenter_weight_vars[k] for k in range(len(barycenter)) for i in range(len(w_list)))
        self.model.addConstrs(self.transport_vars.sum(i, '*', len(barycenter)) <= zeta_a / (1 - zeta_a) for i in range(len(w_list)))
        self.model.addConstr(self.max_weight == gurobipy.max_(self.barycenter_weight_vars))
        pass
    
    def get_barycenter(self):    
        self.model.optimize()
        # b.setObjective(c.prod({i:d[i] for i in c}))
        self.barycenter_weight = np.array([self.barycenter_weight_vars[i].X for i in range(self.barycenter.shape[0])])
        self.transport = [np.array([[self.transport_vars[i, j, k].X for k in range(len(self.barycenter) + 1)] for j in range(len(self.point_lists[i]) + 1)]) for i in range(len(self.w_list))]
        return self.barycenter_weight, self.transport, self.model.ObjVal
    
    def update_barycenter_place(self, barycenter=None, w_list=None, epsilon=None):
        if self.barycenter is None or not barycenter is None and len(barycenter) != len(self.barycenter):
            self.__init__(self.point_lists, barycenter, self.u_list, self.w_list, self.zeta_a, self.zeta_b)
            return self.get_barycenter()
        if not barycenter is None:
            self.barycenter = barycenter
        if not epsilon is None:
            self.epsilon = epsilon
        if not w_list is None:
            self.w_list = w_list
        
        # robust_cost = np.zeros((len(self.point_lists), self.max_point_num + 1, self.barycenter.shape[0] + 1))
        for i in range(len(self.point_lists)):
            point_list = self.point_lists[i]    
            self.robust_cost[i][:len(point_list), :-1] = dist(point_list, self.barycenter) * self.w_list[i]
        
        # update objective function
        # self.model.setObjective(self.transport_vars.prod({i:self.robust_cost[i] for i in self.transport_vars}) + self.max_weight * self.epsilon)
        for i in self.transport_vars:
            self.transport_vars[i].Obj = self.robust_cost[i]
        self.model.update()
        return self.get_barycenter()
    




