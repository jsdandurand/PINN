import sys

from typing import Any, Dict, List, Optional, Tuple
import torch
from collections import OrderedDict

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from RKPINNModel import PhysicsInformedNN, Source
import argparse
import json
import yaml
from tqdm import tqdm
from RKPINNutils.minADE import pre_estimate


np.random.seed(1234)


def u(x, y, downscale_factor=1):
    if isinstance(x, np.ndarray):
        return np.ones_like(x) / downscale_factor * 3
    else:
        return torch.ones_like(x) / downscale_factor * 3

def v(x, y, downscale_factor=1):
    if isinstance(x, np.ndarray):
        return np.ones_like(y) / downscale_factor * 0
    else:
        return torch.ones_like(y) / downscale_factor * 0
    
def apply_downscale_factor(args, downscale_factor):
    # check if args is a dict
    if isinstance(args, dict):
        args['lb'] = [l / downscale_factor for l in args['lb']]
        args['ub'] = [u / downscale_factor for u in args['ub']]
        args['source'] = [s / downscale_factor for s in args['source']]
        args['K'] = args['K'] / (downscale_factor ** 2)
        args['source_size'] = args['source_size'] / downscale_factor
        args['mesh_density'] = args['mesh_density'] * downscale_factor
    else:
        # Downscale domain bounds
        args.lb = [l / downscale_factor for l in args.lb]
        args.ub = [u / downscale_factor for u in args.ub]
        # Downscale source
        args.source = [s / downscale_factor for s in args.source]
        args.K = args.K / (downscale_factor ** 2)
        args.source_size = args.source_size / downscale_factor
        args.mesh_density = args.mesh_density * downscale_factor

    return args
def C_init(x, y):
    # A0 = 2
    # sigma = 0.1
    # x_shift = -0.25
    # y_shift = 0
    # C = A0 * np.exp(-((x - x_shift) ** 2 + (y - y_shift) ** 2) / (sigma ** 2))

    return np.zeros_like(x)

def add_source_approx_data(args, X_0, C, plot=True):
    output_dir = args.output_dir
    # Experiment name
    exp_name = args.exp_name
    # combine output directory and experiment name
    output_dir = os.path.join(output_dir, exp_name)

    lb = np.array(args.lb)
    ub = np.array(args.ub)
    # Add source approx data
    source = np.array(args.source)
    cov = np.diag([args.source_size, args.source_size]) / 100
    source_data = np.random.multivariate_normal(source, cov, int(len(X_0) * args.source_approx_data_ratio))
    # Remove points outside domain
    source_data = source_data[(source_data[:, 0] >= lb[0]) & (source_data[:, 0] <= ub[0]) & (source_data[:, 1] >= lb[1]) & (source_data[:, 1] <= ub[1])]
    X_0 = np.vstack((X_0, source_data))
    C = np.hstack((C, C_init(source_data[:, 0], source_data[:, 1])))

    # Plot initial data and velocity field
    # Get length to height ratio
    ratio = (ub[1] - lb[1]) / (ub[0] - lb[0])
    if plot:
        # Create plot
        plt.figure(figsize=(7 + 3, 7 * ratio + 3))
        plt.scatter(source_data[:, 0], source_data[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Initial data and velocity field')
        # Plot source using a circle
        plt.scatter(source[0], source[1], c='r', marker='o', label='Source')
        # Make plot square
        plt.gca().set_aspect('equal', adjustable='box')
        # Save plot
        plt.savefig(os.path.join(output_dir, 'added_data_points.png'))

    return X_0, C

def init_data_and_config(args, plot=True, save_args=True):

    # Output directory
    output_dir = args.output_dir
    # Experiment name
    exp_name = args.exp_name
    # combine output directory and experiment name
    output_dir = os.path.join(output_dir, exp_name)
    # Create output directory if it does not exist

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if save_args:
        # Save args
        args_path = os.path.join(output_dir, 'args.json')
        with open(args_path, 'w') as f:
            json.dump(vars(args), f)
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Parse arguments

    # Downscale factor
    downscale_factor = args.downscale_factor
    # Apply downscale factor
    args = apply_downscale_factor(args, downscale_factor)

    # ADE Params
    ADE_params = [u, v, args.K]
    ADE_params = np.array(ADE_params) # u, v, K

    # Domain bounds
    lb = np.array(args.lb)
    ub = np.array(args.ub)
    mesh_density = args.mesh_density

    source = np.array(args.source)
    if args.source_middle:
        source = (lb + ub) / 2

    # RK parameters
    n_seconds = args.n_seconds
    dt = args.dt
    q = args.q

    layers = args.layers + [q + 1] # Add output layer

    initial_dim_x = int((mesh_density) * (ub[0] - lb[0]))
    initial_dim_y = int((mesh_density) * (ub[1] - lb[1]))
    boundary_dim_x = int(mesh_density * (ub[0] - lb[0]))
    boundary_dim_y = int(mesh_density * (ub[1] - lb[1]))

    # Initial data
    x_zero = np.linspace(lb[0], ub[0], initial_dim_x)
    y_zero = np.linspace(lb[1], ub[1], initial_dim_y)
    x_mesh_zero, y_mesh_zero = np.meshgrid(x_zero, y_zero)
    x_zero = x_mesh_zero.flatten()
    y_zero = y_mesh_zero.flatten()


    C = C_init(x_zero, y_zero)
    X_0 = np.hstack((x_zero.reshape(-1, 1), y_zero.reshape(-1, 1)))


    # Boundary data
    # x low
    x_x_low = np.ones(boundary_dim_x) * lb[0]
    y_x_low = np.linspace(lb[1], ub[1], boundary_dim_x)

    # x high
    x_x_high = np.ones(boundary_dim_x) * ub[0]
    y_x_high = np.linspace(lb[1], ub[1], boundary_dim_x)

    # y low
    x_y_low = np.linspace(lb[0], ub[0], boundary_dim_y)
    y_y_low = np.ones(boundary_dim_y) * lb[1]

    # y high
    x_y_high = np.linspace(lb[0], ub[0], boundary_dim_y)
    y_y_high = np.ones(boundary_dim_y) * ub[1]

    X_b = np.vstack(
        (
            np.hstack((x_x_low.reshape(-1, 1), y_x_low.reshape(-1, 1))),
            np.hstack((x_x_high.reshape(-1, 1), y_x_high.reshape(-1, 1))),
            np.hstack((x_y_low.reshape(-1, 1), y_y_low.reshape(-1, 1))),
            np.hstack((x_y_high.reshape(-1, 1), y_y_high.reshape(-1, 1)))
        )
    )

    if plot:
        # Plot initial data and velocity field
        # Get length to height ratio
        ratio = (ub[1] - lb[1]) / (ub[0] - lb[0]) 
        # Create plot
        plt.figure(figsize=(7 + 3, 7 * ratio + 3))
        plt.scatter(X_0[:, 0], X_0[:, 1], c=C, cmap='viridis')
        # Streamplot
        x = np.linspace(lb[0], ub[0], 100)
        y = np.linspace(lb[1], ub[1], 100)
        X, Y = np.meshgrid(x, y)
        U = u(X, Y)
        V = v(X, Y)
        plt.streamplot(X, Y, U, V, color='k', density = 1 * ratio)
        cbar = plt.colorbar()
        cbar.set_label('Concentration')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Initial data and velocity field')
        # Plot source using a circleadfast2/dt5downscale1-wind0310xK
        plt.scatter(source[0], source[1], c='r', marker='o', label='Source')


        # Make plot square
        plt.gca().set_aspect('equal', adjustable='box')
        # Save plot
        plt.savefig(os.path.join(output_dir, 'initial_data.png'))

    # Add more data points around source
    if args.source_approx_data:
        X_0, C = add_source_approx_data(args, X_0, C, plot)
    # Print number of training points
    print('Number of initial and boundary data points: ', len(X_0) + len(X_b))

    return X_0, X_b, C, ADE_params, device, output_dir

def train_models(X_0, X_b, C, ADE_params, device, output_dir, args) -> Tuple[List[float], List[float]]:
    # Load config
    n_seconds = args.n_seconds
    dt = args.dt
    q = args.q
    layers = args.layers
    lb = np.array(args.lb)
    ub = np.array(args.ub)
    source = Source(args.source, args.source_size, args.source_concentration)
    epochs_adam = args.epochs_adam
    epochs_lbfgs = args.epochs_lbfgs

    # Models path
    models_path = os.path.join(output_dir, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    print('Models will be saved to: ', models_path)
    # Train models
    last_model = None
    model_train_times = []
    losses = []
    n_models = int(n_seconds / dt)
    loss_threshold = args.loss_threshold
    print('Loss threshold: ', loss_threshold)

    with tqdm(total=n_models, desc = 'Models Trained') as pbar:
        for time_step in range(n_models):
            start_time = time.time()
            model = PhysicsInformedNN(X_0, X_b, C, layers, lb, ub, ADE_params, dt, q, loss_threshold, source,
                                      device,
                                      epochs_adam,
                                      epochs_lbfgs,
                                      args.downscale_factor)

            # set the parameters of dnn to be the same as the previous model
            if time_step > 0:
                model.dnn.load_state_dict(last_model.dnn.state_dict())

            model.train()

            # Delete last model
            if time_step > 0:
                del last_model
            last_model = model

            # Predict initial data for next time step
            _, C = model.predict(X_0)
            C = C[:, -1] # Get last time step

            # Clear cache
            torch.cuda.empty_cache()
            model_train_times.append(round(time.time() - start_time, 2))

            # Save model to models_path
            model_path = os.path.join(models_path, 'model_%d.pth' % time_step)
            torch.save(model.dnn.state_dict(), model_path)
            
            loss = model.loss_func().item()
            losses.append(loss)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'Loss': loss})

    
    return model_train_times, losses



def parse_args():

    ap = argparse.ArgumentParser()

    # RK PARAMS
    rk_group = ap.add_argument_group('RK parameters')
    rk_group.add_argument('--n_seconds', type=int, default=1, help='Number of seconds to simulate')
    rk_group.add_argument('--dt', type=int, default=1, help='Time step')
    rk_group.add_argument('--q', type=int, default=500, help='Number of stages in the Runge-Kutta method')

    # ADE PARAMS
    ade_group = ap.add_argument_group('ADE parameters')
    ade_group.add_argument('--K', type=float, default=1e-3, help='Diffusion coefficient')
    ade_group.add_argument('--downscale_factor', type=int, default=1, help='Downscale factor')

    # INITIAL AND BOUNDARY DATA PARAMS

    data_group = ap.add_argument_group('Initial and boundary data')
    # UPPER AND LOWER BOUNDS
    data_group.add_argument('--lb', type=float, nargs=2, default=[-5, -5], help='Lower bound of the domain')
    data_group.add_argument('--ub', type=float, nargs=2, default=[5, 5], help='Upper bound of the domain')
    # SOURCE
    data_group.add_argument('--source', type=float, nargs=2, default=[0, 0], help='Source location')
    data_group.add_argument('--source_middle', type=bool, default=False, help='Source location in the middle of the domain')
    data_group.add_argument('--source_size', type=float, default=0.2032 / 2, help='Source size (radius)')
    data_group.add_argument('--source_concentration', type=float, default=0.4594 * 5, help='Source concentration')
    data_group.add_argument('--source_approx_data', type=bool, default=False, help='Whether to add source approx data')
    data_group.add_argument('--source_approx_data_ratio', type=float, default=0.1, help='Source approx data ratio to initial data')
    # INITIAL AND BOUNDARY DATA RATIO
    data_group.add_argument('--mesh_density', type=float, default=7, help='Mesh density (points per unit length)')

    # MODEL PARAMS
    model_group = ap.add_argument_group('Model parameters')
    # LAYERS
    model_group.add_argument('--layers', type=int, nargs='+', default=[2, 50, 50, 50, 50, 50, 50], help='Number of neurons in each layer, excluding output layer')
    # Epochs adam and lbfgs
    model_group.add_argument('--epochs_adam', type=int, default=1000, help='Number of epochs for Adam optimizer')
    model_group.add_argument('--epochs_lbfgs', type=int, default=5000, help='Number of epochs for L-BFGS optimizer')
    model_group.add_argument('--loss_threshold', type=int, default=1000, help='Loss threshold')
    # OTHER PARAMS
    other_group = ap.add_argument_group('Other parameters')
    # Output directory
    other_group.add_argument('--output_dir', type=str, default='results', help='Output directory')
    # Experiment name
    other_group.add_argument('--exp_name', type=str, default='ADE-RK-PINN', help='Experiment name')

    # PRE ESTIMATE PARAMS
    pre_estimate_group = ap.add_argument_group('Pre-estimate parameters')
    # Do pre-estimate
    pre_estimate_group.add_argument('--pre_estimate', type=bool, default=False, help='Whether to pre-estimate')
    # Time step
    pre_estimate_group.add_argument('--time_step', type=int, default=2.5e-2, help='Time step to pre-estimate')
    # n steps
    pre_estimate_group.add_argument('--n_steps', type=int, default=40, help='Number of steps to pre-estimate')



    return ap.parse_args()

def main():
    args = parse_args()
    # Print args
    print("============================== ARGS ==============================")
    print(args)
    print("==================================================================")

    # Initialize data and config
    X_0, X_b, C, ADE_params, device, output_dir = init_data_and_config(args)

    # Print args used
    print('============================== ARGS USED ==============================')
    print(args)
    print("=======================================================================")
    # Print device used
    print('Device: ', device)

    # Pre estimate
    if args.pre_estimate:
        C = pre_estimate(X_0, C, ADE_params, args)

    # Train models
    model_train_times, losses = train_models(X_0, X_b, C, ADE_params, device, output_dir, args)

    # Print time taken to train each model
    print('============================== RESULTS ==============================')
    print('Model train times:')
    print(model_train_times)
    
    # Print total time taken to train all models
    total_time = sum(model_train_times)
    print('Total time taken to train all models: ', total_time)

    # Print losses rounded to 4 decimal places
    print('Losses:')
    print([round(loss, 4) for loss in losses])
    print("=====================================================================")

    # Save times and losses
    times_losses = np.vstack((model_train_times, losses)).T
    times_losses_path = os.path.join(output_dir, 'times_losses.csv')
    np.savetxt(times_losses_path, times_losses, delimiter=',', header='Time (s), Loss')



if __name__ == '__main__':
    main()