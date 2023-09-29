import numpy as np
import tensorflow as tf
import h5py
from utils import evaluation_utils as e_utils

if __name__ == "__main__":
    
    data_dir = "../../data/cerebro_data"
    hr_filename = "patient3-postOp_HR.h5"

    lr_filename = "patient3-postOp_LR.h5"
    
    prediction_dir = "../predictions/4DFlowNet"
    prediction_filename = "patient3-postOp_SR.h5"
    
    ground_truth_file = f"{data_dir}/{hr_filename}"
    prediction_file = f"{prediction_dir}/{prediction_filename}"
    lr_file = f"{data_dir}/{lr_filename}"

    peak_flow_idx = 34

    with h5py.File(lr_file, mode = 'r') as hf:
        u_lr = np.asarray(hf['u'][peak_flow_idx])
        v_lr = np.asarray(hf['v'][peak_flow_idx])
        w_lr = np.asarray(hf['w'][peak_flow_idx])

    with h5py.File(prediction_file, mode = 'r') as hf:
        u_sr = np.asarray(hf['u'][peak_flow_idx])
        v_sr = np.asarray(hf['v'][peak_flow_idx])
        w_sr = np.asarray(hf['w'][peak_flow_idx])
    
    with h5py.File(ground_truth_file, 'r') as hf:
        u_hr = np.asarray(hf['u'][peak_flow_idx])
        v_hr = np.asarray(hf['v'][peak_flow_idx])
        w_hr = np.asarray(hf['w'][peak_flow_idx])

        mask = np.asarray(hf['mask'])

    # x_slice = 126

    # SR = u_sr[x_slice:x_slice+2, 40:60, 60:80]
    # HR = u_hr[x_slice:x_slice+2, 40:60, 60:80]
    # LR = u_lr[x_slice//2:x_slice//2+2, 20:30, 30:40]
    # MSK = mask[x_slice:x_slice+2, 40:60, 60:80]

    y_slice = 50

    SR = v_sr[110:130, y_slice:y_slice+2, 100:120]
    HR = v_hr[110:130, y_slice:y_slice+2, 100:120]
    LR = v_lr[55:65, y_slice//2:y_slice//2+2, 50:60] 
    MSK = mask[110:130, y_slice:y_slice+2, 100:120]

    # z_slice = 120

    # SR = w_sr[60:80, 60:80, z_slice:z_slice+2]
    # HR = w_hr[60:80, 60:80, z_slice:z_slice+2]
    # LR = w_lr[30:40, 30:40, z_slice//2:z_slice//2+2]
    # MSK = mask[60:80, 60:80, z_slice:z_slice+2]

    # z_slice = 100

    # SR = u_sr[98:122, 60:84, z_slice:z_slice+2]
    # HR = u_hr[98:122, 60:84, z_slice:z_slice+2]
    # LR = u_lr[49:61, 30:42, z_slice//2:z_slice//2+2]
    # MSK = mask[98:122, 60:84, z_slice:z_slice+2]

    bound, core = e_utils._create_boundary_and_core_masks(MSK, 0.1, 'voxels')

    absdiff = np.absolute(SR - HR)

    k = 0.5
    absdiff = np.clip(absdiff/(2*k) + 0.5, 0.0, 1.0)

    m = 1.0
    SR = np.clip(SR/(2*m) + 0.5, 0.0, 1.0)
    HR = np.clip(HR/(2*m) + 0.5, 0.0, 1.0)
    LR = np.clip(LR/(2*m) + 0.5, 0.0, 1.0)
    MSK = MSK*0.5+0.5

    bound = bound*0.5+0.5
    core = core*0.5+0.5

    e_utils.generate_gif_volume(SR, axis=1, save_as=f'{prediction_file[:-3]}_zoom3', norm=False, format='PNG', scaling=4)
    # e_utils.generate_gif_volume(HR, axis=1, save_as=f'{prediction_file[:-3]}_zoom3HR', norm=False, format='PNG', scaling=4)
    # e_utils.generate_gif_volume(LR, axis=1, save_as=f'{prediction_file[:-3]}_zoom3LR', norm=False, format='PNG', scaling=4)
    # e_utils.generate_gif_volume(MSK, axis=0, save_as=f'{prediction_file[:-3]}_zoomMask', norm=False, format='PNG', scaling=4)

    # e_utils.generate_gif_volume(bound, axis=0, save_as=f'{prediction_file[:-3]}_zoomBound', norm=False, format='PNG', scaling=4)
    # e_utils.generate_gif_volume(core, axis=0, save_as=f'{prediction_file[:-3]}_zoomCore', norm=False, format='PNG', scaling=4)

    e_utils.generate_gif_volume(absdiff, axis=1, save_as=f'{prediction_file[:-3]}_zoom3Diff', norm=False, format='PNG', scaling=4)
