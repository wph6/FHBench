from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
import copy
import numpy as np
def train(model,train_loader,optimizer,scheduler,loss_func):
    running_loss = 0.0
    all_preds = []
    all_targets = []
    model.train()
    for data, label in tqdm(train_loader, ncols=60, desc="train", unit="b", leave=None):
        data, label = data.cuda().float(), label.cuda().long()
        optimizer.zero_grad()
        pred = model.forward(data)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        all_preds.extend(pred.argmax(dim=1).cpu().numpy())
        all_targets.extend(label.cpu().numpy())

    scheduler.step()
    loss = running_loss / len(train_loader)
    acc = accuracy_score(all_targets, all_preds)
    return loss,acc

def train_prox(args,model,server_model,train_loader,optimizer,scheduler,loss_func):
    running_loss = 0.0
    all_preds = []
    all_targets = []
    model.train()
    for data, label in tqdm(train_loader, ncols=60, desc="train", unit="b", leave=None):
        data, label = data.cuda().float(), label.cuda().long()
        optimizer.zero_grad()
        pred = model.forward(data)
        loss = loss_func(pred, label)
        
        w_diff = torch.tensor(0.).cuda()
        for w, w_t in zip(server_model.parameters(), model.parameters()):
            w_diff += torch.norm(w - w_t) ** 2
        loss += args.mu / 2. * w_diff
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        all_preds.extend(pred.argmax(dim=1).cpu().numpy())
        all_targets.extend(label.cpu().numpy())

    scheduler.step()
    loss = running_loss / len(train_loader)
    acc = accuracy_score(all_targets, all_preds)
    return loss,acc

def train_APFL(args,client_model,server_dict,train_loader,optimizer,scheduler,server_optimizer,server_scheduler,loss_func):
    def alpha_update(model_local, model_personal,alpha, eta):
        grad_alpha = 0
        for l_params, p_params in zip(model_local.parameters(), model_personal.parameters()):
            if p_params.grad is not None and l_params.grad is not None:
                dif = p_params.data - l_params.data
                grad = alpha * p_params.grad.data + (1-alpha)*l_params.grad.data
                grad_alpha += dif.view(-1).T.dot(grad.view(-1))
        
        grad_alpha += 0.02 * alpha
        alpha_n = alpha - eta*grad_alpha
        alpha_n = np.clip(alpha_n.item(),0.0,1.0)
        return alpha_n

    server_model = copy.deepcopy(client_model)
    server_model.load_state_dict(server_dict)
    
    client_model.train()
    server_model.train()

    for data, label in tqdm(train_loader, ncols=60, desc="train", unit="b", leave=None):
        data, label = data.cuda().float(), label.cuda().long()
        

        pred = client_model(data)
        loss = loss_func(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_server = server_model(data)
        loss = loss_func(pred_server, label)
        server_optimizer.zero_grad()
        loss.backward()
        server_optimizer.step()

    scheduler.step()
    server_scheduler.step()
    alpha = alpha_update(client_model,server_model,args.alpha_APFL,args.lr)
    
    for lp, p in zip(client_model.parameters(), server_model.parameters()):
        lp.data = (1 - alpha) * p + alpha * lp

    param = copy.deepcopy(server_model.state_dict())
    return param

def train_APPLE(args,model,params,ps,client,lamda,train_loader,optimizer,scheduler,loss_func):
    model.train()
    for data, label in tqdm(train_loader, ncols=60, desc="train", unit="b", leave=None):
        data, label = data.cuda().float(), label.cuda().long()
        optimizer.zero_grad()
        pred = model.forward(data)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
    scheduler.step()

    for param_c, param in zip(params,[p for p in model.parameters() if p.requires_grad]):
        if param.grad is not None:
            param_c.data = param_c - args.lr * param.grad * ps[client]
    p0 = 1 / args.n_clients
    for cid in range(args.n_clients):
        cnt = 0
        p_grad = 0
        for param_c, param in zip(params,[p for p in model.parameters() if p.requires_grad]):
            if param.grad is not None:
                p_grad += torch.mean(param.grad * param_c).item()
                cnt += 1
        p_grad = p_grad / cnt
        p_grad = p_grad + lamda * args.mu * (ps[cid] - p0)
        ps[cid] = ps[cid] - args.dr_learning_rate * p_grad

    param = {name: param.clone() for name, param in model.named_parameters() if param.requires_grad}
    return param

def adaptive_agg_ALA(args,model,server_model,weights,train_loader,round,loss_func):

    # obtain the references of the parameters
    params_g = [p for p in server_model.parameters() if p.requires_grad]
    params = [p for p in model.parameters() if p.requires_grad]

    # deactivate ALA at the 1st communication iteration
    if torch.sum(params_g[0] - params[0]) == 0:
        return
    
    # preserve all the updates in the lower layers
    for param, param_g in zip(params[:-args.layer_idx], params_g[:-args.layer_idx]):
        param.data = param_g.data.clone()

    # temp local model only for weight learning
    model_t = copy.deepcopy(model)
    params_t = [p for p in model_t.parameters() if p.requires_grad]

    # only consider higher layers
    params_p = params[-args.layer_idx:]
    params_gp = params_g[-args.layer_idx:]
    params_tp = params_t[-args.layer_idx:]

    # frozen the lower layers to reduce computational cost in Pytorch
    for param in params_t[:-args.layer_idx]:
        param.requires_grad = False
    # used to obtain the gradient of higher layers
    # no need to use optimizer.step(), so lr=0
    optimizer = torch.optim.SGD(params_tp, lr=0)    

    for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,weights):
        param_t.data = param + (param_g - param) * weight

    losses = []
    cnt = 0 
    while True:
        for data, label in tqdm(train_loader, ncols=60, desc="train", unit="b", leave=None):
            data, label = data.cuda().float(), label.cuda().long()
            optimizer.zero_grad()
            pred = model.forward(data)
            loss = loss_func(pred, label)
            loss.backward()

            # update weight in this batch
            for param_t, param, param_g, weight in zip(params_tp, params_p,params_gp, weights):
                if param_t.grad is not None:
                    weight.data = torch.clamp(
                        weight - args.eta * (param_t.grad * (param_g - param)), 0, 1)
            # update temp local model in this batch
            for param_t, param, param_g, weight in zip(params_tp, params_p,params_gp, weights):
                param_t.data = param + (param_g - param) * weight
        losses.append(loss.item())
        cnt += 1

        if round != 1:
            break
        if len(losses) > args.num_pre_loss and np.std(losses[-args.num_pre_loss:]) < args.threshold:
            break

    for param, param_t in zip(params_p, params_tp):
        param.data = param_t.data.clone()

def train_SCAFFOLD(args,model,server_model,global_c,client_c,train_loader,optimizer,scheduler,loss_func):

    model.train()
    for data, label in tqdm(train_loader, ncols=60, desc="train", unit="b", leave=None):
        data, label = data.cuda().float(), label.cuda().long()
        optimizer.zero_grad()
        pred = model.forward(data)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step(global_c,client_c)
    scheduler.step()

    num_batches = len(train_loader)
    server_params = [p for p in server_model.parameters() if p.requires_grad]
    model_params = [p for p in model.parameters() if p.requires_grad]

    for ci, c, x, yi in zip(client_c, global_c, server_params, model_params):
        ci.data = ci - c + 1/num_batches/args.epochs/args.lr * (x - yi)
    delta_y = []
    delta_c = []    
    for c, x, yi in zip(global_c, server_params, model_params):
        delta_y.append(yi - x)
        delta_c.append(- c + 1/num_batches/args.epochs/args.lr * (x - yi))
    return delta_y,delta_c

@torch.no_grad()
def evaluate(model,loader,loss_func):
    loss_all = 0
    all_preds = []
    all_targets = []
    model.eval()
    for data, label in tqdm(loader, ncols=60, unit="b", leave=None):
        data, label = data.cuda(), label.cuda()
        pred = model.forward(data)
        loss = loss_func(pred, label)
        loss_all += loss.item()

        all_preds.extend(pred.argmax(dim=1).cpu().numpy())
        all_targets.extend(label.cpu().numpy())

    loss = loss_all / len(loader)
    acc = accuracy_score(all_targets, all_preds)
    return loss,acc