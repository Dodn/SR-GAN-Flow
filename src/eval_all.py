import numpy as np
import tensorflow as tf
import h5py
from Network import loss_utils
from utils import evaluation_utils as e_utils

if __name__ == "__main__":
    
    data_dir = "../../data/cerebro_data"
    hr_filename = "patient3-postOp_HR.h5"

    lr_filename = "patient3-postOp_LR.h5"
    
    prediction_dir = "../predictions"
    #pred_names = ['4DFlowNet', 'GAN_2D', 'GAN_Ftest', 'GAN_M', 'GAN_MB', 'GAN_MU', 'GAN_X', 'GAN_ESR', 'GAN_ESR2', 'GAN_ESR3']
    # pred_names = ['4DFlowNet',
    #               'GAN_2D',
    #               'GAN',      'GAN/m1',      'GAN/m2',      'GAN/m3',
    #               'GAN_M',    'GAN_M/m1',    'GAN_M/m2',    'GAN_M/m3',
    #               'GAN_MB',   'GAN_MB/m1',   'GAN_MB/m2',   'GAN_MB/m3',
    #               'GAN_MU',   'GAN_MU/m1',   'GAN_MU/m2',   'GAN_MU/m3',
    #               'GAN_ESR',  'GAN_ESR/m1',  'GAN_ESR/m2',  'GAN_ESR/m3',
    #               'GAN_ESRB', 'GAN_ESRB/m1', 'GAN_ESRB/m2', 'GAN_ESRB/m3',
    #               ]
    pred_names = ['4DFlowNet',   '4DFlowNet/m1',   '4DFlowNet/m2',   '4DFlowNet/m3',
                  'GAN_2D',      'GAN_2D/m1',      'GAN_2D/m2',      'GAN_2D/m3',
                 ]
    prediction_filename = "patient3-postOp_SR.h5"
    
    ground_truth_file = f"{data_dir}/{hr_filename}"
    prediction_files = [f"{prediction_dir}/{p}/{prediction_filename}" for p in pred_names]
    lr_file = f"{data_dir}/{lr_filename}"

    peak_flow_idx = 34

    with h5py.File(ground_truth_file, 'r') as hf:
        u_hr = np.asarray(hf['u'])
        v_hr = np.asarray(hf['v'])
        w_hr = np.asarray(hf['w'])

        T = len(hf.get("u"))
        mask = np.asarray(hf['mask'])
        nf_mask = 1.0 - mask
        boundary_mask, core_mask = e_utils._create_boundary_and_core_masks(mask, 0.1, 'voxels')
        # assert np.all(core_mask + boundary_mask == mask)

        X,Y,Z = mask.shape
        cov_a = np.sum(mask)/(X*Y*Z)
        cov_b = np.sum(boundary_mask)/(X*Y*Z)
        cov_c = np.sum(core_mask)/(X*Y*Z)
        ratio_b = np.sum(boundary_mask)/np.sum(mask)
        ratio_c = np.sum(core_mask)/np.sum(mask)

        print(' ')
        print(f'Coverage: {100*cov_a}%')
        print(f'Boundary --- cov: {100*cov_b}%, ratio: {100*ratio_b}%')
        print(f'Core --- cov: {100*cov_c}%, ratio: {100*ratio_c}%')

    for i, pred_file in enumerate(prediction_files):
        name = pred_names[i]
        print(' ')
        print('-'*25)
        print(name)
        print('-'*25)

        with h5py.File(pred_file, mode = 'r') as pf:
            u_sr = np.asarray(pf['u'])
            v_sr = np.asarray(pf['v'])
            w_sr = np.asarray(pf['w'])

        rel_err = np.zeros((T,3))
        abs_err = np.zeros((T,4))
        rmse = np.zeros((T,4))

        Ks = np.zeros((T,3,3))
        Ms = np.zeros((T,3,3))
        Rs = np.zeros((T,3,3))

        for t in range(T):
            rel_err[t,0] = (e_utils.calculate_relative_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], mask))
            rel_err[t,1] = (e_utils.calculate_relative_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
            rel_err[t,2] = (e_utils.calculate_relative_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))

            abs_err[t,0] = (e_utils.calculate_absolute_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], mask))
            abs_err[t,1] = (e_utils.calculate_absolute_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
            abs_err[t,2] = (e_utils.calculate_absolute_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))
            abs_err[t,3] = (e_utils.calculate_absolute_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], nf_mask))

            rmse[t,0] = (e_utils.calculate_rmse(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], mask))
            rmse[t,1] = (e_utils.calculate_rmse(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
            rmse[t,2] = (e_utils.calculate_rmse(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))
            rmse[t,3] = (e_utils.calculate_rmse(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], nf_mask))

            Ks[t][0][0], Ms[t][0][0], Rs[t][0][0] = e_utils.linreg(u_sr[t], u_hr[t], mask)
            Ks[t][1][0], Ms[t][1][0], Rs[t][1][0] = e_utils.linreg(v_sr[t], v_hr[t], mask)
            Ks[t][2][0], Ms[t][2][0], Rs[t][2][0] = e_utils.linreg(w_sr[t], w_hr[t], mask)

            Ks[t][0][1], Ms[t][0][1], Rs[t][0][1] = e_utils.linreg(u_sr[t], u_hr[t], boundary_mask)
            Ks[t][1][1], Ms[t][1][1], Rs[t][1][1] = e_utils.linreg(v_sr[t], v_hr[t], boundary_mask)
            Ks[t][2][1], Ms[t][2][1], Rs[t][2][1] = e_utils.linreg(w_sr[t], w_hr[t], boundary_mask)

            Ks[t][0][2], Ms[t][0][2], Rs[t][0][2] = e_utils.linreg(u_sr[t], u_hr[t], core_mask)
            Ks[t][1][2], Ms[t][1][2], Rs[t][1][2] = e_utils.linreg(v_sr[t], v_hr[t], core_mask)
            Ks[t][2][2], Ms[t][2][2], Rs[t][2][2] = e_utils.linreg(w_sr[t], w_hr[t], core_mask)
        
        print('Total avg')
        rel_err_tot = np.mean(rel_err, axis=0)
        print(f'Relative error [Fluid] {rel_err_tot[0]}')
        print(f'Relative error [Bound] {rel_err_tot[1]}')
        print(f'Relative error [Core ] {rel_err_tot[2]}')

        abs_err_tot = np.mean(abs_err, axis=0)
        print(f'Absolute error [Fluid] {abs_err_tot[0]}')
        print(f'Absolute error [Bound] {abs_err_tot[1]}')
        print(f'Absolute error [Core ] {abs_err_tot[2]}')
        print(f'Absolute error [Non-F] {abs_err_tot[3]}')

        rmse_tot = np.mean(rmse, axis=0)
        print(f'R.M.S.   error [Fluid] {rmse_tot[0]}')
        print(f'R.M.S.   error [Bound] {rmse_tot[1]}')
        print(f'R.M.S.   error [Core ] {rmse_tot[2]}')
        print(f'R.M.S.   error [Non-F] {rmse_tot[3]}')

        print('-  '*9)
        print('Peak Flow')

        print(f'Relative error [Fluid] {rel_err[peak_flow_idx][0]}')
        print(f'Relative error [Bound] {rel_err[peak_flow_idx][1]}')
        print(f'Relative error [Core ] {rel_err[peak_flow_idx][2]}')

        print(f'Absolute error [Fluid] {abs_err[peak_flow_idx][0]}')
        print(f'Absolute error [Bound] {abs_err[peak_flow_idx][1]}')
        print(f'Absolute error [Core ] {abs_err[peak_flow_idx][2]}')
        print(f'Absolute error [Non-F] {abs_err[peak_flow_idx][3]}')

        print(f'R.M.S.   error [Fluid] {rmse[peak_flow_idx][0]}')
        print(f'R.M.S.   error [Bound] {rmse[peak_flow_idx][1]}')
        print(f'R.M.S.   error [Core ] {rmse[peak_flow_idx][2]}')
        print(f'R.M.S.   error [Non-F] {rmse[peak_flow_idx][3]}')

        print(' ')
        print(f'U [Fluid] k: {Ks[peak_flow_idx][0][0]} \t m: {Ms[peak_flow_idx][0][0]} \t r^2: {Rs[peak_flow_idx][0][0]}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][0][1]} \t m: {Ms[peak_flow_idx][0][1]} \t r^2: {Rs[peak_flow_idx][0][1]}')
        print(f'  [Core ] k: {Ks[peak_flow_idx][0][2]} \t m: {Ms[peak_flow_idx][0][2]} \t r^2: {Rs[peak_flow_idx][0][2]}')

        print(' ')
        print(f'V [Fluid] k: {Ks[peak_flow_idx][1][0]} \t m: {Ms[peak_flow_idx][1][0]} \t r^2: {Rs[peak_flow_idx][1][0]}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][1][1]} \t m: {Ms[peak_flow_idx][1][1]} \t r^2: {Rs[peak_flow_idx][1][1]}')
        print(f'  [Core ] k: {Ks[peak_flow_idx][1][2]} \t m: {Ms[peak_flow_idx][1][2]} \t r^2: {Rs[peak_flow_idx][1][2]}')

        print(' ')
        print(f'W [Fluid] k: {Ks[peak_flow_idx][2][0]} \t m: {Ms[peak_flow_idx][2][0]} \t r^2: {Rs[peak_flow_idx][2][0]}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][2][1]} \t m: {Ms[peak_flow_idx][2][1]} \t r^2: {Rs[peak_flow_idx][2][1]}')
        print(f'  [Core ] k: {Ks[peak_flow_idx][2][2]} \t m: {Ms[peak_flow_idx][2][2]} \t r^2: {Rs[peak_flow_idx][2][2]}')
