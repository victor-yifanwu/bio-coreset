import torch
from tqdm import tqdm
from torch.func import functional_call, vmap, jacrev

class ParamInf():
    def __init__(self, model, loss_fn, device, specific_layer=False, target_layers=[],
                 *args, **kwargs) -> None:
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        
        self.model.eval()
        
        if not specific_layer:
            self.params = {k: v.detach() for k, v in model.named_parameters()}
        else:
            self.params = {k: v.detach() for k, v in model.named_parameters() if k in target_layers}
        self.diag_fisher = {k: torch.zeros_like(v).view(1, -1) for k, v in self.params.items()}
        self.inv_diag_fisher = {k: torch.zeros_like(v).view(1, -1) for k, v in self.params.items()}
        
    def _fnet_single(self, params, x, y):
        model_params = {k: v.detach() for k, v in self.model.named_parameters()}
        for k, v in params.items():
            model_params[k] = v
        y_hat = functional_call(self.model, model_params, (x,))
        y_hat = y_hat.view(-1, y_hat.size(-1)) 
        y = y.view(-1) 
        loss = self.loss_fn(y_hat, y)
        return loss
    
    def _calc_grad(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        grad = vmap(jacrev(self._fnet_single), in_dims=(None, 0, 0))(self.params, x, y)
        grad = {k: v.flatten(1) for k, v in grad.items()}
        return grad
    
    # 实际是做一个累加
    def sigma_fisher(self, x, y, num): 
        x = x.to(self.device)
        y = y.to(self.device)

        grad_batch = self._calc_grad(x, y)
        for k in self.params.keys():
            self.diag_fisher[k].add_(grad_batch[k].square_().sum(dim=0, keepdim=True) / num)

    def finalize_inverse_fisher(self):
        for k in self.params.keys():
            self.inv_diag_fisher[k] = 1 / (self.diag_fisher[k] + 1e-8)
        return self.inv_diag_fisher
    
    def set_inverse_fisher(self, inv_diag_fisher):
        self.inv_diag_fisher = inv_diag_fisher

    def calc_influence(self, x_tr, y_tr):
        
        x_tr = x_tr.to(self.device)
        y_tr = y_tr.to(self.device)

        grad_tr = self._calc_grad(x_tr, y_tr)
        influence_score = 0
        for k in self.params.keys():
            influence_score += - (grad_tr[k].square() * self.inv_diag_fisher[k]).sum(dim=1)

        return influence_score
    
    def calc_parameter_influence(self, x_tr, y_tr):
        
        x_tr = x_tr.to(self.device)
        y_tr = y_tr.to(self.device)

        grad_tr = self._calc_grad(x_tr, y_tr)
        for k in self.params.keys():
            p_i= - (self.inv_diag_fisher[k] * grad_tr[k])

        return p_i