import sys

from typing import Any, Dict, List, Optional, Tuple
import torch
from collections import OrderedDict

import os
from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import time
from RKPINNModel import PhysicsInformedNN, DNN
from ADERKtrain import apply_downscale_factor
import argparse
import json
import yaml
from tqdm import tqdm
import imageio

def validate_args(args, args_train: Dict[str, Any]) -> None:
    lb, ub = args_train['lb'], args_train['ub']
    print(lb, ub)
    # Check if line plot start and end are within bounds
    if args.line_plot:
        x1, y1, x2, y2 = args.line_plot_start_end
        assert lb[0] <= x1 <= ub[0] and lb[0] <= x2 <= ub[0], "Line plot start and end x values must be within bounds"
        assert lb[1] <= y1 <= ub[1] and lb[1] <= y2 <= ub[1], "Line plot start and end y values must be within bounds"



def load_IRK_params(q, dt = 1):
    tmp = np.float32(np.loadtxt('Butcher_IRK%d.txt' % (q), ndmin = 2))
    IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))
    c = tmp[q**2+q:]
    b = IRK_weights[-1] 
    A = IRK_weights[:-1]
    return A, b, c

def check_args_or_die(args):
    assert os.path.exists(args.config), "Config file not found"
    assert not args.compare_gt or os.path.exists(args.gt_data), "Ground truth data not found"

def load_models(config_dir, args_train, device) -> List[PhysicsInformedNN]:
    layers = args_train['layers']
    q = args_train['q']
    layers = layers + [q + 1]

    # Get model dir
    model_dir = os.path.join(config_dir, 'models')

    # Load models
    models = []
    model_dir_list = os.listdir(model_dir)
    # Sort models by time. Nodel are saved as 'model_time.pth'
    model_dir_list = sorted(model_dir_list, key=lambda x: float(x.split('_')[1].split('.pth')[0]))
    print(model_dir_list)

    for model_name in model_dir_list:
        model_path = os.path.join(model_dir, model_name)
        model = DNN(layers)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)
        models.append(model)
    
    return models

def get_resolution_if_gt_data(args):
    if args.gt_data != "":
        # Load ground truth data
        gt = np.load(os.path.join(args.gt_data, '0.00.npy'))
        res = gt.shape[0]
    else:
        res = args.resolution
    return res

def plot_line_plot(models, A, b, c, args_train, args, device):
    res = get_resolution_if_gt_data(args)
    dt = args_train['dt']
    images_per_model = dt * args.fps 
    all_images = []
    lb, ub = args_train['lb'], args_train['ub']
    ratio = (ub[1] - lb[1]) / (ub[0] - lb[0]) 
    output_path = os.path.dirname(args.config)
    output_path_full = os.path.join(output_path, 'line_plot_test')
    output_path_full_csv = os.path.join(output_path, 'line_plot_test_csv')
    if not os.path.exists(output_path_full):
        os.makedirs(output_path_full)
        
    if not os.path.exists(output_path_full_csv):
        os.makedirs(output_path_full_csv)
    n_seconds = args_train['n_seconds']
    pe_time = args_train['time_step'] * args_train['n_steps'] if args_train['pre_estimate'] else 0

    c = torch.tensor(c).to(device)
    # Create meshgrid
    x = torch.linspace(lb[0], ub[0], res)
    y = torch.linspace(lb[1], ub[1], res)
    x, y = torch.meshgrid(x, y)
    x = x.reshape(-1).unsqueeze(1).to(device)
    y = y.reshape(-1).unsqueeze(1).to(device)
    x_star = torch.cat((x, y), 1)

    # Convert x, y to numpy
    x = x.cpu().numpy()
    y = y.cpu().numpy()

    x1, y1, x2, y2 = args.line_plot_start_end
    # Get true x1, y1, x2, y2 based on resolution
    x_line = np.linspace(lb[0], ub[0], res)
    y_line = np.linspace(lb[1], ub[1], res)
    x1_idx = np.argmin(np.abs(x_line - x1))
    y1_idx = np.argmin(np.abs(y_line - y1))
    x2_idx = np.argmin(np.abs(x_line - x2))
    y2_idx = np.argmin(np.abs(y_line - y2))
    x_idx_line = np.arange(x1_idx, x2_idx + 1)
    y_idx_line = np.arange(y1_idx, y2_idx + 1)
    # Adjust length of shorter line to match longer line by stretching
    if len(x_idx_line) > len(y_idx_line):
        y_idx_line = np.linspace(y1_idx, y2_idx, len(x_idx_line)).astype(int)
    elif len(y_idx_line) > len(x_idx_line):
        x_idx_line = np.linspace(x1_idx, x2_idx, len(y_idx_line)).astype(int)
    

    nplots = 2 if args.compare_gt else 1

    for i, model in enumerate(models):
        start_time = dt * i

        indices = []
        times = np.linspace(0, dt, images_per_model + 1)
        actual_times = times + start_time + pe_time

        # Round to 1 decimal place
        times = np.round(times, 2)
        normalized_times = times / dt
        for t in normalized_times:
            indices.append(torch.argmin(torch.abs(c - t)))

        out = model.forward(x_star) # shape res^2 x q+1

        for j, idx in enumerate(indices):
            C = out[: , idx].detach().cpu().numpy().reshape(res, res)

            if args.gt_data != "":
                gt = np.load(os.path.join(args.gt_data, '%.2f.npy' % actual_times[j])).T
                error = np.mean((gt - C)**2)

            # Get line plot
            C_line = C[x_idx_line, y_idx_line]

            path = os.path.join(output_path_full, 'line_plot_time_%.2f.png' % actual_times[j])
            
            # Save as csv, C_line vs x_line and y_line
            np.savetxt(os.path.join(output_path_full_csv, 'line_plot_time_%.2f.csv' % actual_times[j]), np.vstack((x_line[x_idx_line], y_line[y_idx_line], C_line)).T, delimiter=',', header='x,y,C', comments='')

            # Line plot
            plt.figure(figsize=(5 + 3 , 5 * nplots * ratio + 3))
            plt.subplot(nplots, 1, 1)
            plt.plot(np.linspace(0, 1, len(C_line)), C_line)
            plt.title('Predicted concentration:')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.gca().set_aspect('equal', adjustable='box')

            if not args.compare_gt:
                plt.suptitle('Time: %.2f' % actual_times[j])
            else:
                # Line plot for ground truth
                gt_line = gt[x_idx_line, y_idx_line]
                plt.subplot(2, 1, 2)
                plt.plot(np.linspace(0, 1, len(gt_line)), gt_line)
                plt.title('Ground truth')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.gca().set_aspect('equal', adjustable='box')

            plt.tight_layout()

            # Add to list of images
            plt.savefig(path)
            plt.close()

            all_images.append(path)

    images = []
    for filename in all_images:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(output_path_full, 'line_plot_concentration.gif'), images, duration=n_seconds)




def eval_models(models, A, b, c, args_train, args, device):
    
    res = get_resolution_if_gt_data(args)
    dt = args_train['dt']
    images_per_model = dt * args.fps 
    all_images = []
    lb, ub = args_train['lb'], args_train['ub']
    ratio = (ub[1] - lb[1]) / (ub[0] - lb[0]) 
    output_path = os.path.dirname(args.config)
    output_path_full = os.path.join(output_path, 'test')
    if not os.path.exists(output_path_full):
        os.makedirs(output_path_full)
    n_seconds = args_train['n_seconds']
    pe_time = args_train['time_step'] * args_train['n_steps'] if args_train['pre_estimate'] else 0

    c = torch.tensor(c).to(device)
    # Create meshgrid
    x = torch.linspace(lb[0], ub[0], res)
    y = torch.linspace(lb[1], ub[1], res)
    x, y = torch.meshgrid(x, y)
    x = x.reshape(-1).unsqueeze(1).to(device)
    y = y.reshape(-1).unsqueeze(1).to(device)
    x_star = torch.cat((x, y), 1)

    # Convert x, y to numpy
    x = x.cpu().numpy()
    y = y.cpu().numpy()

    vmin, vmax = 0, 0.3
    cmap = 'gist_heat'
    nplots = 2 if args.compare_gt else 1

    for i, model in enumerate(models):
        start_time = dt * i

        indices = []
        times = np.linspace(0, dt, images_per_model + 1)
        actual_times = times + start_time + pe_time

        # Round to 1 decimal place
        times = np.round(times, 2)
        normalized_times = times / dt
        for t in normalized_times:
            indices.append(torch.argmin(torch.abs(c - t)))

        out = model.forward(x_star) # shape res^2 x q+1

        for j, idx in enumerate(indices):
            C = out[: , idx].detach().cpu().numpy().reshape(res, res)

            if args.gt_data != "":
                gt = np.load(os.path.join(args.gt_data, '%.2f.npy' % actual_times[j])).T
                error = np.mean((gt - C)**2)
    
            path = os.path.join(output_path_full, 'time_%.2f.png' % actual_times[j])

            # Scatter plot
            plt.figure(figsize=(5 + 3 , 5 * nplots * ratio + 3))
            plt.subplot(nplots, 1, 1)
            plt.scatter(x, y, c=C, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.title('Predicted concentration:')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.gca().set_aspect('equal', adjustable='box') 

            if not args.compare_gt:
                plt.suptitle('Time: %.2f' % actual_times[j])
            else:
                # Scatter plot for ground truth
                plt.subplot(2, 1, 2)
                plt.scatter(x, y, c=gt, cmap=cmap, vmin=vmin, vmax=vmax)
                plt.colorbar()
                plt.title('Ground truth')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.gca().set_aspect('equal', adjustable='box')

                # Add error and time to title
                plt.suptitle('Error: %.5f, Time: %.2f' % (error, actual_times[j]))

            plt.tight_layout()

            # Add to list of images
            plt.savefig(path)
            plt.close()

            # Append to all images
            all_images.append(path)

    images = []
    for filename in all_images:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(output_path_full, 'concentration.gif'), images, duration=n_seconds)


def parse_args():
    parser = argparse.ArgumentParser(description="Test ADE-RK model")
    parser.add_argument("--config", type=str, default="results/circle-dt5-2/args.json", help="Path to the config file")
    parser.add_argument("--resolution", "-res", type=int, default=300, help="Resolution of the plot")
    parser.add_argument("--compare_gt", type=bool, default=False, help="Compare with ground truth data")
    parser.add_argument("--gt_data", type=str, default="", help="Path to the ground truth data")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second")
    parser.add_argument("--line_plot", type=bool, default=False, help="Plot line plot")
    parser.add_argument("--line_plot_start_end", nargs=4, type=float, default=[0, 0, 55, 0], help="Start and end of line plot (x1, y1, x2, y2)")
    return parser.parse_args()

def main():

    args = parse_args()
    check_args_or_die(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract directory of config file
    config_dir = os.path.dirname(args.config)
    
    # Load args from args.json
    with open(args.config, "r") as f:
        args_train = json.load(f)
    args_train = apply_downscale_factor(args_train, args_train['downscale_factor'])
    x1, y1, x2, y2 = args.line_plot_start_end
    # Divide by downscale factor
    downscale_factor = args_train['downscale_factor']
    x1, y1, x2, y2 = x1 / downscale_factor, y1 / downscale_factor, x2 / downscale_factor, y2 / downscale_factor
    args.line_plot_start_end = [x1, y1, x2, y2]
    validate_args(args, args_train)
    
    # Load models
    models = load_models(config_dir, args_train, device)
    A, b, c = load_IRK_params(args_train['q'], args_train['dt'])

    # Evaluate models
    eval_models(models, A, b, c, args_train, args, device)

    # Plot line plot
    if args.line_plot:
        plot_line_plot(models, A, b, c, args_train, args, device)

if __name__ == "__main__":
    main()







