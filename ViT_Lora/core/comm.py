import torch
import copy
import numpy as np
from torch.optim import Optimizer

def cal_sim_mat(args,param_list,model_momentum=0.5):
    client_num = len(param_list)
    weight_m = np.zeros((client_num, client_num))
    num_layers = len(param_list[0])
    if args.layers_part == "first_half":
        layers_to_include = list(range(num_layers // 2))
    elif args.layers_part == "second_half":
        layers_to_include = list(range(num_layers // 2, num_layers))
    else:
        layers_to_include = list(range(num_layers))
        
    for i in range(client_num):
        for j in range(client_num):
            if i == j:
                weight_m[i, j] = 0
            else:
                l2_distance = [
                    np.linalg.norm((param_list[i][layer] - param_list[j][layer]).cpu().float().numpy())
                    for layer in layers_to_include
                ]
                avg_l2_dis = np.mean(l2_distance)
                if l2_distance == 0:
                    weight_m[i, j] = 1e12
                else:
                    weight_m[i, j] = 1 / avg_l2_dis
    weight_sums = np.sum(weight_m, axis=1)
    weight_m = weight_m / weight_sums[:, np.newaxis] 
    weight_m = weight_m * (1 - model_momentum)
    for i in range(client_num):
        weight_m[i, i] = model_momentum
    return weight_m



def global_aggregate(args, server_model, models, client_weights,APFL_params,APPLE_params,pss,dys,dcs,global_c,weight_m):
    client_num = len(models)
    with torch.no_grad():
        if args.fed_alg.lower() == 'fedavg' or args.fed_alg.lower() == 'fedprox':
            for key in server_model.state_dict().keys():
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        elif args.fed_alg == 'EPFL':
            tmpmodels=[]
            for i in range(client_num):
                tmpmodels.append(copy.deepcopy(models[i]))
            for cl in range(args.n_clients):
                for key in server_model.state_dict().keys():
                    if 'linear_a' in key:
                        temp = torch.zeros_like(server_model.state_dict()[key])  
                        for client_idx in range(args.n_clients):
                            temp += weight_m[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                        models[cl].state_dict()[key].data.copy_(temp)

        elif args.fed_alg == 'APFL':
            for key in server_model.state_dict().keys():
                temp = torch.zeros_like(server_model.state_dict()[key])  
                for client_idx in range(args.n_clients):
                    temp += client_weights[client_idx] * APFL_params[client_idx][key]
                server_model.state_dict()[key].data.copy_(temp)

        elif args.fed_alg == 'APPLE':
            for cl in range(args.n_clients):
                for key in APPLE_params[0].keys():
                    temp = torch.zeros_like(APPLE_params[0][key])  
                    for client_idx in range(args.n_clients):
                        temp += pss[cl,client_idx] * APPLE_params[client_idx][key]
                    models[cl].state_dict()[key].data.copy_(temp)

        elif args.fed_alg == 'fedALA':
            for key in server_model.state_dict().keys():
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)

        elif args.fed_alg == 'SCAFFOLD':
            for client_idx in range(args.n_clients):
                dy,dc = dys[client_idx],dcs[client_idx]
                for server_param, client_param in zip([p for p in server_model.parameters() if p.requires_grad],dy):
                    server_param.data += client_param.data.clone() / args.n_clients * args.server_learning_rate
                for server_c, client_param in zip(global_c, dc):
                    server_c.data += client_param.data.clone() / args.n_clients
                    
        else:
            raise ValueError(f"Aggregation strategy {args.fed_alg} not supported")


    return server_model, models

class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cs, client_cs):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                if p.grad is not None:
                    p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])