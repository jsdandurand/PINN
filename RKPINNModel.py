
import sys

import torch
from collections import OrderedDict

from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

np.random.seed(1234)

class Source:
    def __init__(self, source_loc, source_size, source_concentration):
        self.source_loc = source_loc
        self.source_size = source_size
        self.source_concentration = source_concentration

    def __call__(self, x, y):
        source_val = self.source_concentration
        source = torch.zeros_like(x)
        source[(x >= self.source_loc[0]-self.source_size) &
                (x <= self.source_loc[0]+self.source_size) &
                (y >= self.source_loc[1]-self.source_size) & 
                (y <= self.source_loc[1]+self.source_size)] = source_val
        return source

# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = torch.nn.Tanh
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

        
    def forward(self, x):
        out = self.layers(x)
        return out
    
# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X_0, X_b, C, layers, lb, ub, ADE_params, dt, q, loss_threshold, source,
                 device = 'cpu', 
                 epochs_adam = 1000,
                 epochs_lbfgs = 5000,
                 downscale_factor = 1.0,):
        
        self.device = device
        self.loss_threshold = loss_threshold
        self.downscale_factor = downscale_factor
        self.source = source
        
        # Runge-Kutta parameters
        self.dt = dt
        self.q = max(q, 1)
        tmp = np.float32(np.loadtxt('Butcher_IRK%d.txt' % (q), ndmin = 2))
        IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))
        c = tmp[q**2+q:]
        b = IRK_weights[-1]
        A = IRK_weights[:-1]
            
        # Turn to tensors
        self.A = torch.tensor(A, dtype=torch.float32).to(device)
        self.b = torch.tensor(b, dtype=torch.float32).to(device) 
        self.c = torch.tensor(c, dtype=torch.float32).to(device)
        
        # ADE parameters
        self.ADE_params = ADE_params
        
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
        
        # data
        self.x_0 = torch.tensor(X_0[:, 0:1], requires_grad=True).float().to(device)
        self.y_0 = torch.tensor(X_0[:, 1:2], requires_grad=True).float().to(device)
        self.C = torch.tensor(C).float().to(device).unsqueeze(1)
        
        # boundary data points
        self.x_b = torch.tensor(X_b[:, 0:1], requires_grad=True).float().to(device)
        self.y_b = torch.tensor(X_b[:, 1:2], requires_grad=True).float().to(device)
        
        self.layers = layers + [q + 1]

        # deep neural networks
        self.dnn = DNN(self.layers).to(device)
        
        # optimizers: using the same settings
        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        self.epochs_adam = epochs_adam
        self.adam_complete = False
        self.epochs_lbfgs = epochs_lbfgs
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=self.epochs_lbfgs, 
            max_eval=self.epochs_lbfgs, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )

        self.iter = 0

        self.pbar_adam = tqdm(total=self.epochs_adam, desc='Adam', file=sys.stdout)
        self.pbar_lbfgs = tqdm(total=self.epochs_lbfgs, desc='L-BFGS', file=sys.stdout)
        
    def net_C1(self, x, y):  
        c = self.dnn(torch.cat([x, y], dim=1))
        return c
    
    def gradient(self, dy: torch.Tensor,
                dx,
                ones_like_tensor= None,
                create_graph: bool = True):
        """Compute the gradient of a tensor `dy` with respect to another tensor `dx`.
        :param dy: The tensor to compute the gradient for.
        :param dx: The tensor with respect to which the gradient is computed.
        :param ones_like_tensor: A tensor with the same shape as `dy`, used for creating the gradient (default is None).
        :param create_graph: Whether to create a computational graph for higher-order gradients (default is True).
        :return: The gradient of `dy` with respect to `dx`.
        """
        if ones_like_tensor is None:
            grad_outputs = [torch.ones_like(dy)]
        else:
            grad_outputs = ones_like_tensor

        if isinstance(dx, torch.Tensor):
            dx = [dx]

        dy_dx = torch.autograd.grad(
            [dy],
            dx,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=False,
        )
        
        grads = [grad if grad is not None else torch.zeros_like(dx[i]) for i, grad in enumerate(dy_dx)]
        return grads    
    
    def fwd_gradient(self, dy: torch.Tensor, 
                    dx,
                    create_graph: bool = True):
        """Compute the forward gradient of a tensor `dy` with respect to another tensor `dx`.

        :param dy: The tensor to compute the forward gradient for.
        :param dx: The tensor with respect to which the forward gradient is computed.
        :return: The forward gradient of `dy` with respect to `dx`.
        """
            
        if isinstance(dx, torch.Tensor):
            dx = [dx]
        
        ones_like = [torch.ones_like(dy).requires_grad_(True)]
            
        grads = self.gradient(dy, dx, ones_like)

        fwd_grads = []
        for i, grad in enumerate(grads):
            
            ones_like_ = ones_like[0]
            assert ones_like_ is not None
            
            fwd_grad = self.gradient(grad, ones_like_, create_graph=create_graph)[0]
            
            if isinstance(fwd_grad, torch.Tensor):
                fwd_grads.append(fwd_grad)

        return fwd_grads  
      
    def diff_op(self, x, y):
        u, v, K = self.ADE_params
        
        out = self.net_C1(x, y)
        Cx = self.fwd_gradient(out, x)[0]
        Cy = self.fwd_gradient(out, y)[0]
        Cxx = self.fwd_gradient(Cx, x)[0]
        Cyy = self.fwd_gradient(Cy, y)[0]
        return -1 * K * (Cxx + Cyy) + u(x, y, self.downscale_factor) * Cx + v(x, y, self.downscale_factor) * Cy  - self.source.__call__(x, y).reshape(-1, 1).to(self.device)
    
    def net_C0(self, x, y):
        """ The pytorch autograd version of calculating residual """
        C_preds = self.net_C1(x, y)
        f_C = self.diff_op(x, y)
            # f_C has shape batch_size x q+1
        C_i = C_preds[:, :self.q] + torch.matmul(f_C[:, :self.q], self.A.T) * self.dt # batch_size x q
        C_q_plus_1 = C_preds[:, self.q] + torch.matmul(f_C[:, :self.q], self.b) * self.dt 
        C_q_plus_1.unsqueeze_(1) # batch_size x 1
        return torch.cat((C_i, C_q_plus_1), 1) # batch_size x q+1
        
    
    def loss_func(self):
        self.optimizer.zero_grad()
        
        C0_pred = self.net_C0(self.x_0, self.y_0)
        loss_C0 = torch.sum(torch.sum((C0_pred - self.C)**2))
        
        C1_pred = self.net_C1(self.x_b, self.y_b)
        loss_C1 = torch.sum(torch.sum(C1_pred**2))
        
        loss = loss_C0 + loss_C1
        loss.backward()
        if self.adam_complete:
            self.iter += 1
            # update every 1 percent
            if self.iter % (self.epochs_lbfgs // 100) == 0:
                self.pbar_lbfgs.update(self.epochs_lbfgs // 100)
                self.pbar_lbfgs.set_postfix({'Loss': loss.item()})
        return loss
    
    def failure_informed_adaptive_sampling(self):
        # Update X_0 based on the current model
        C0_pred = self.net_C0(self.x_0, self.y_0)
        loss_C0 = torch.sum(torch.sum((C0_pred - self.C)**2))
        loss_C0.backward()
        self.x_0.grad = torch.abs(self.x_0.grad)
        self.y_0.grad = torch.abs(self.y_0.grad)
        self.x_0 = self.x_0 + 0.1 * self.x_0.grad
        self.y_0 = self.y_0 + 0.1 * self.y_0.grad
        self.x_0 = self.x_0.detach().requires_grad_()
        self.y_0 = self.y_0.detach().requires_grad_()

    def train(self):
        self.dnn.train()

        # RUN ADAM
        for epoch in range(self.epochs_adam):
            self.optimizer_adam.zero_grad()
            loss = self.loss_func()
            self.optimizer_adam.step()
            if epoch % (self.epochs_adam // 100) == 0:
                self.pbar_adam.update(self.epochs_adam // 100)
                self.pbar_adam.set_postfix({'Loss': loss.item()})

        self.adam_complete = True
        
        # RUN LBFGS
        self.optimizer.step(self.loss_func)
        loss = self.loss_func()

        # Continue running LBFGS until loss is below threshold
        # Less adam epochs are needed
        self.epochs_adam = int(self.epochs_adam / 5)
        while loss > self.loss_threshold:
            # Run Adam for a few epochs
            self.adam_complete = False
            self.pbar_adam.reset()
            for epoch in range(self.epochs_adam):
                self.optimizer_adam.zero_grad()
                loss = self.loss_func()
                self.optimizer_adam.step()

                # Progress bar
                if epoch % (self.epochs_adam // 100) == 0:
                    self.pbar_adam.update(self.epochs_adam // 100)
                    self. pbar_adam.set_postfix({'Loss': loss.item()})

            self.adam_complete = True
            self.optimizer.max_iter = 1000
            self.pbar_lbfgs.reset()
            self.pbar_lbfgs.total = 1000
            self.optimizer.step(self.loss_func) 
            loss = self.loss_func()
        
        # Loss based adaptive sampling
        #self.failure_informed_adaptive_sampling()

        self.pbar_lbfgs.close()
        self.pbar_adam.close()

            
    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        self.dnn.eval()
        C0 = self.net_C0(x, y)
        C1 = self.net_C1(x, y)
        C0 = C0.detach().cpu().numpy()
        C1 = C1.detach().cpu().numpy()
        return C0, C1
