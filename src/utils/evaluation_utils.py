import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import numpy as np
from scipy import stats
from skimage import morphology

# Crop HR data (if necessary). Expects 3D shapes.
def crop(hr, pred):
    # We assume that if there is a mismatch it's because SR is smaller than HR.
    crop = np.asarray(hr.shape) - np.asarray(pred.shape)
    hr = hr[crop[0]//2:-crop[0]//2,:,:] if crop[0] else hr
    hr = hr[:, crop[1]//2:-crop[1]//2,:] if crop[1] else hr
    hr = hr[:, :, crop[2]//2:-crop[2]//2] if crop[2] else hr
    return hr
 

def get_slice_values(body, idx, axis='x'):
    if axis=='x':
        vals = body[idx, :, :]
    elif axis=='y':
        vals = body[:, idx, :]
    elif axis=='z':
        vals = body[:,:,idx]
    else:
        print("Error: x, y, z are available axes")
        return
    return vals

# Available vel_dirs are u, v, w.
def slice(ax, file, frame, idx, vel_dir='u', axis='x'):
    with h5py.File(file, mode = 'r' ) as hf:
        body = np.asarray(hf[vel_dir][frame])
        sliced = get_slice_values(body, idx, axis)
    ax.imshow(sliced, interpolation='nearest', cmap='viridis', origin='lower')
                
    
def generate_slice_comp(files, frame, lr_idx, fig_nr, vel_dir='u', axis='x'):
    plt.figure(fig_nr)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.set_title("LR")
    ax2.set_title("HR")
    ax3.set_title("SR")

    lr_file = files[0]

    hr_file = files[1]
    hr_idx = lr_idx*2

    sr_file = files[2]

    # LR plot
    slice(ax1, lr_file, frame, lr_idx, vel_dir, axis)

    # HR plot
    slice(ax2, hr_file, frame, hr_idx, vel_dir, axis)

    # SR plot
    slice(ax3, sr_file, frame, hr_idx, vel_dir, axis)
    plt.savefig("test.png")
    return fig_nr + 1


    

def plot_relative_mean_error(relative_mean_error, N, save_file,fig_nr):
    print(f"Plotting relative mean error...")
    plt.figure(fig_nr)
    if N == 1:
        plt.scatter(N, relative_mean_error)
    else:
        plt.plot(np.arange(N), relative_mean_error)
    plt.xlabel("Frame")
    plt.ylabel("Relative error (%)")
    plt.title("Relative speed error")
    plt.savefig(f"{save_file[:-3]}_RME.png")
    return fig_nr + 1

def plot_mean_speed_old(mean_speed, N, save_file, fig_nr):
    plt.figure(fig_nr)
    fig_nr += 1
    if N == 1:
        plt.scatter(N, mean_speed)
    else:
        plt.plot(np.arange(N), mean_speed)
    plt.xlabel("Frame")
    plt.ylabel("Avg. speed (cm/s)")
    plt.savefig(f"{save_file[:-3]}_speed.png")
    return fig_nr

def plot_mean_speed(mean_speed, N, save_file, fig_nr):
    print("Plotting average speed...")
    plt.figure(fig_nr)
    fig, ax = plt.subplots()

    colors = ['r', 'g', 'b', 'y']
    labels = ['$\mathregular{|V|}$', '$\mathregular{V_x}$', '$\mathregular{V_y}$', '$\mathregular{V_z}$']
    for i in range(4):
        ax.plot(tf.range(N), mean_speed[:, i], color=colors[i], label=labels[i])

    ax.set_xlabel("Frames")
    ax.set_ylabel("Avg. speed (cm/s)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_file[:-3]}_speed.png")
    plt.show()
    return fig_nr + 1

def _reg_stats(hr_vals, sr_vals):
    reg = stats.linregress(hr_vals, sr_vals)
    x = np.array([-10, 10]) # Start, End point for the regression slope lines
    if reg.intercept < 0.0:
        reg_stats = f'$y = {reg.slope:.3f}x {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$'
    else:
        reg_stats = f'$y = {reg.slope:.3f}x + {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$'
    plt.plot(x, reg.intercept + reg.slope*x, 'k', linestyle='--', alpha=0.3)
    return reg_stats

def _plot_linear_regression(fig_nr, dimension, boundary_hr_vals, boundary_sr_vals, core_hr_vals, core_sr_vals):
    plt.figure(fig_nr)
    
    # Set limits and ticks
    xlim = ylim = 1.0
    plt.xlim(-xlim,xlim); plt.ylim(-ylim,ylim); plt.xticks([-xlim, xlim]); plt.yticks([-ylim, ylim])
    
    plt.scatter(core_hr_vals, core_sr_vals, s=0.8, c=["black"])
    plt.scatter(boundary_hr_vals, boundary_sr_vals, s=0.2, c=["red"])
    
    boundary_reg_stats = _reg_stats(boundary_hr_vals, boundary_sr_vals)
    plt.text(-xlim/2, ylim/2, boundary_reg_stats, horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='red')
    core_reg_stats = _reg_stats(core_hr_vals, core_sr_vals)
    plt.text(xlim/2, -ylim/2, core_reg_stats, horizontalalignment='center', verticalalignment='top', fontsize=10, color="black")
    
    # Set title and labels
    plt.title(f"Correlation in V_{dimension}"); plt.xlabel("V_HR [m/s]"); plt.ylabel("V_SR [m/s]")

def _create_boundary_and_core_masks(binary_mask, cut_off, cut_off_type='voxels'):
    if cut_off < 1:
        selem = None
    elif cut_off_type == 'voxels':
        selem = np.ones((cut_off, cut_off, cut_off))
    elif cut_off_type == 'percentage':
        cut_off = int(np.ceil(cut_off * np.min(binary_mask.shape)))
        selem = np.ones((cut_off, cut_off, cut_off))
    else:
        raise ValueError("Invalid cut_off_type, choose 'voxels' or 'percentage'")
    
    core_mask = morphology.binary_erosion(binary_mask, selem)
    boundary_mask = np.logical_xor(binary_mask, core_mask)
    return boundary_mask.astype(np.int32), core_mask.astype(np.int32)

def _sample_hrsr(ground_truth_file, prediction_file, mask, peak_flow_idx, ratio):
    # Use mask to find interesting samples
    sample_pot = np.where(mask == 1)
    rng = np.random.default_rng()

    # Sample <ratio> samples
    sample_idx = rng.choice(len(sample_pot[0]), replace=False, size=(int(len(sample_pot[0])*ratio)))

    # Get indexes
    x_idx = sample_pot[0][sample_idx]
    y_idx = sample_pot[1][sample_idx]
    z_idx = sample_pot[2][sample_idx]

    with h5py.File(prediction_file, mode = 'r' ) as hf:
        sr_u = np.asarray(hf['u'][peak_flow_idx])
        sr_u_vals = sr_u[x_idx, y_idx, z_idx]
        sr_v = np.asarray(hf['v'][peak_flow_idx])
        sr_v_vals = sr_v[x_idx, y_idx, z_idx]
        sr_w = np.asarray(hf['w'][peak_flow_idx])
        sr_w_vals = sr_w[x_idx, y_idx, z_idx]
        
    with h5py.File(ground_truth_file, mode = 'r' ) as hf:
        # Get velocity values in all directions
        hr_u = crop(np.asarray(hf['u'][peak_flow_idx]), sr_u)
        hr_u_vals = hr_u[x_idx, y_idx, z_idx]
        hr_v = crop(np.asarray(hf['v'][peak_flow_idx]), sr_v)
        hr_v_vals = hr_v[x_idx, y_idx, z_idx]
        hr_w = crop(np.asarray(hf['w'][peak_flow_idx]), sr_w)
        hr_w_vals = hr_w[x_idx, y_idx, z_idx]
        
        
    return [hr_u_vals, hr_v_vals, hr_w_vals], [sr_u_vals, sr_v_vals, sr_w_vals]
    

def draw_reg_line(ground_truth_file, prediction_file, peak_flow_idx, binary_mask, fig_nr):
    """ Plot a linear regression between HR and predicted SR in peak flow frame """
    #
    # Parameters
    #
    fig_nr += 1
    ratio = 0.1
    
    boundary_mask, core_mask = _create_boundary_and_core_masks(binary_mask, 0.1, 'voxels',) 
    
    boundary_hr, boundary_sr = _sample_hrsr(ground_truth_file, prediction_file, boundary_mask, peak_flow_idx, ratio)
    core_hr, core_sr = _sample_hrsr(ground_truth_file, prediction_file, core_mask, peak_flow_idx, ratio)

    print(f"Plotting regression lines...")
    
    _plot_linear_regression(fig_nr, "x", boundary_hr[0], boundary_sr[0], core_hr[0], core_sr[0])
    plt.savefig(f"{prediction_file[:-3]}_LRXplot.png")
    _plot_linear_regression(fig_nr+1, "y", boundary_hr[1], boundary_sr[1], core_hr[1], core_sr[1])
    plt.savefig(f"{prediction_file[:-3]}_LRYplot.png")
    _plot_linear_regression(fig_nr+2, "z", boundary_hr[2], boundary_sr[2], core_hr[2], core_sr[2])
    plt.savefig(f"{prediction_file[:-3]}_LRZplot.png")
    return fig_nr+3
