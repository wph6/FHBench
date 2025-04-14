import os
from momentfm import MOMENTPipeline
import torch
from tqdm import tqdm 
from peft import LoraConfig, get_peft_model
import argparse
import numpy as np
from timeseries_lora.core.traineval import *
import math
from datautil.prepare_data import get_data
import copy
import wandb
from core.comm import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #training parameters
    parser.add_argument('--rank', default=8, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=16, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--reduction', type=str, default='concat', help='reduction method for MOMENT embeddings, choose from mean or max')
    parser.add_argument('--dataset', type=str, default='ADFTD',choices=['PTBXL', 'ADFTD'], )
 
    # Fed arguments
    parser.add_argument("--fed_alg", type=str, default="fedALA")
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--n_clients", type=int, default=20)
    parser.add_argument('--partition_data', type=str,default='non_iid_dirichlet', help='partition data way')
    parser.add_argument('--non_iid_alpha', type=float,default=0.1, help='data split for label shift')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument('--datapercent', type=float,default=0.2, help='data percent to use')
    
    # fedprox    
    parser.add_argument('--mu', type=float, default=1e-3,help='The hyper parameter for fedprox')
    parser.add_argument('--model_momentum', type=float, default=0.5,help='model_momentum for agg')
    # APFL 
    parser.add_argument("--alpha_APFL", type=float, default=1.0)
    # APPLE
    parser.add_argument('-drlr', "--dr_learning_rate", type=float, default=1e-3)
    # ALA
    parser.add_argument("--layer_idx", type=int, default=26)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--num_pre_loss", type=float, default=10)
    parser.add_argument("--threshold", type=float, default=0.1)
    # SCAFFOLD
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # EPFL
    parser.add_argument("--model_momentum", type=float, default=0.5)
    parser.add_argument("--layers_part", type=str, default="all", choices=["first_half", "second_half", "all"], help="Part of the model layers to include.")
    
    args = parser.parse_args()
    args.random_state = np.random.RandomState(2025)
    
    save_name = f'{args.dataset}{args.datapercent}_{args.fed_alg}_rank{args.rank}_alpha{args.alpha}_cl_{args.n_clients}_lr_{args.lr}'
    wandb.init(project="TimesLora",name=save_name)
    
    train_loaders, val_loaders, test_loaders, num_classes, num_channels = get_data(args.dataset)(args)

    model = MOMENTPipeline.from_pretrained(
                                            "AutonLab/MOMENT-1-small", 
                                            model_kwargs={
                                                'task_name': 'classification',
                                                'n_channels': num_channels,
                                                'num_class': num_classes,
                                                'reduction': args.reduction,
                                                'freeze_encoder': False,
                                                'freeze_embedder': False,
                                                'enable_gradient_checkpointing': False , 
                                            })
    model.init()
    criterion = torch.nn.CrossEntropyLoss()  
    
    lora_config = LoraConfig(
                            r=8,
                            lora_alpha=16,
                            target_modules=["q", "v"],
                            lora_dropout=0.05,
                            )
    model = get_peft_model(model, lora_config)
    print('LoRA enabled')
    model.print_trainable_parameters()
    
    server_model = model.cuda()
    client_models = [copy.deepcopy(server_model) for _ in range(args.n_clients)]
    
    if args.fed_alg == 'SCAFFOLD':
        optimizers = [SCAFFOLDOptimizer(params=[p for p in client_models[idx].parameters() if p.requires_grad],
                             lr=args.lr) for idx in range(args.n_clients)]
    else:
        optimizers = [optim.SGD(params=client_models[idx].parameters(),
                                lr=args.lr) for idx in range(args.n_clients)]

    schedulers = [CosineAnnealingLR(optimizer, args.n_rounds, 1e-6) for optimizer in optimizers]
    server_optimizer = optim.SGD(params=server_model.parameters(), lr=args.lr)
    server_scheduler = CosineAnnealingLR(server_optimizer, args.n_rounds, 1e-6)
    
    best_vacc_list, best_tacc_list = [0] * args.n_clients, [0] * args.n_clients
    client_weights = [1 / args.n_clients for _ in range(args.n_clients)]
    
    APFL_params = [[] for i in range(args.n_clients)]

    pss= np.full((args.n_clients, args.n_clients), fill_value=1/args.n_clients)
    APPLE_params = [{name: param.clone().detach() for name, param in server_model.named_parameters() if param.requires_grad}
                        for _ in range(args.n_clients) ]
    ALA_params = [p for p in server_model.parameters() if p.requires_grad][-args.layer_idx:]
    ALA_weights  = [[torch.ones_like(param) for param in ALA_params]
                        for _ in range(args.n_clients)]

    global_c = [torch.zeros_like(param) for param in server_model.parameters() if param.requires_grad] # c in SCAFFOLD
    client_cs = [copy.deepcopy(global_c) for _ in range(args.n_clients)]    # c_i in SCAFFOLD
    delta_c = [copy.deepcopy(global_c) for _ in range(args.n_clients)]    # delta c_i in SCAFFOLD
    dys,dcs = [[] for i in range(args.n_clients)],[[] for i in range(args.n_clients)]
    for round in tqdm(range(args.n_rounds)):
        print(f">> ==================== Round {round + 1} ====================")
        global_dict = server_model.state_dict()
        # APPLE
        all_params = [[param.clone().detach() for param in state_dict.values()] for state_dict in APPLE_params]
        mat_va = [[] for i in range(args.n_clients)]
        for client in range(args.n_clients):
            lamda = (math.cos(round * math.pi / args.n_rounds) + 1) / 2
            for epoch in range(1, args.epochs + 1):
                if args.fed_alg == 'fedprox':
                    train_prox(args,client_models[client],server_model,train_loaders[client],optimizers[client],schedulers[client],criterion)
                elif args.fed_alg == 'APFL':
                    APFL_params[client] = train_APFL(args,client_models[client],global_dict,train_loaders[client],optimizers[client],schedulers[client],server_optimizer,server_scheduler,criterion)
                elif args.fed_alg == 'APPLE':
                    APPLE_params[client] = train_APPLE(args,client_models[client],all_params[client],pss[client],client,lamda,train_loaders[client],optimizers[client],schedulers[client],criterion)   
                elif args.fed_alg == 'fedALA':
                    adaptive_agg_ALA(args,client_models[client],server_model,ALA_weights[client],train_loaders[client],round,criterion)
                    train(args,client_models[client],train_loaders[client],optimizers[client],schedulers[client],criterion)   
                elif args.fed_alg == 'SCAFFOLD':
                    dys[client],dcs[client] = train_SCAFFOLD(args,client_models[client],server_model,global_c,client_cs[client],train_loaders[client],optimizers[client],schedulers[client],criterion)
                else:    
                    train(args,client_models[client],train_loaders[client],optimizers[client],schedulers[client],criterion)
       
            if (round + 1) % 5 == 0:
                train_loss,_ = evaluate(args,client_models[client], train_loaders[client], criterion)
                val_loss,val_acc = evaluate(args,client_models[client], val_loaders[client], criterion)
                wandb.log({
                        f"Training Loss/client_{client}": train_loss,
                        f"Validation Loss/client_{client}": val_loss,
                        f"Validation acc/client_{client}": val_acc,
                        },step = round + 1)
                
                if val_acc > best_vacc_list[client]:
                    best_vacc_list[client] = val_acc
                    _,test_acc = evaluate(args,client_models[client], test_loaders[client], criterion)
                    best_tacc_list[client] = test_acc
            if args.fed_alg == 'EPFL':
                for key in client_models[client].state_dict().keys():
                    if 'lora_B' in key:
                        mat_va[client].append(client_models[client].state_dict()[key])
        weight_m = cal_sim_mat(mat_va)    
        global_aggregate(args,server_model,client_models,client_weights,APFL_params,APPLE_params,pss,dys,dcs,global_c,weight_m)
    
    wandb.log({"client mean acc": np.mean(best_tacc_list)})
    print("Client Mean Accuracy:", np.mean(best_tacc_list))   