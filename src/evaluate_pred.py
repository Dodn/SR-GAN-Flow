import numpy as np
import tensorflow as tf
import h5py
from Network import loss_utils
from utils import evaluation_utils as e_utils

if __name__ == "__main__":
    """ Plot the relative error and mean speed across all frames in the image as well as regression plots
        for the peak flow frame. This evaluates a super resolved h5 file (generated from prediction.py) 
        compared to the ground truth HR image from the dataset. """
    
    data_dir = "../../data/cerebro_data"
    hr_filename = "patient3-postOp_HR.h5"

    lr_filename = "patient3-postOp_LR.h5"
    
    prediction_dir = "../predictions/GAN_ESR"
    prediction_filename = "patient3-postOp_SR.h5"


    # data_dir = "../../data/BRAIN/Healthy/HV06/h5"
    # hr_filename = "HV06.h5"

    # lr_filename = "HV06_LR.h5"
    
    # prediction_dir = "../predictions/invivo/4DFlowNet/Healthy"
    # prediction_filename = "HV06_LR.h5"

    ground_truth_file = f"{data_dir}/{hr_filename}"
    prediction_file = f"{prediction_dir}/{prediction_filename}"
    lr_file = f"{data_dir}/{lr_filename}"
    
    # Parameters
    mask_threshold = 0.6
    
    # Get number of frames and mask
    with h5py.File(prediction_file, mode = 'r') as hf:
        u = tf.convert_to_tensor(hf['u'][0])

        u_sr = np.asarray(hf['u'])
        v_sr = np.asarray(hf['v'])
        w_sr = np.asarray(hf['w'])
    
    with h5py.File(ground_truth_file, 'r') as hf:
        mask = tf.convert_to_tensor(hf['mask'])
        if len(mask.shape) == 3: 
            mask = e_utils.crop(mask, u)[tf.newaxis]
        else:
            mask = e_utils.crop(mask[0], u)[tf.newaxis]
        # Casting excessively because eager tensors won't dynamically cast. 
        binary_mask = tf.cast((tf.cast(mask, dtype=tf.float32) >= mask_threshold), dtype=tf.float32)
        data_count = len(hf.get("u"))

        u_hr = np.asarray(hf['u'])
        v_hr = np.asarray(hf['v'])
        w_hr = np.asarray(hf['w'])

        np_mask = np.asarray(hf['mask'])
        
    rel_err = np.zeros(data_count)
    mean_speed = np.zeros((data_count, 4))

    peak_flow = -1
    peak_flow_idx = -1
    
    for idx in range(data_count):
        # Load the prediction U V W from H5
        with h5py.File(prediction_file, mode = 'r' ) as hf:
            pred_u = tf.convert_to_tensor(hf['u'][idx])[tf.newaxis]
            pred_v = tf.convert_to_tensor(hf['v'][idx])[tf.newaxis]
            pred_w = tf.convert_to_tensor(hf['w'][idx])[tf.newaxis]
            
        # Load the ground truth U V W from H5 and crop if necessary
        with h5py.File(ground_truth_file, mode = 'r' ) as hf:
            hr_u = e_utils.crop(tf.convert_to_tensor(hf['u'][idx]), pred_u[0])[tf.newaxis]
            hr_v = e_utils.crop(tf.convert_to_tensor(hf['v'][idx]), pred_v[0])[tf.newaxis]
            hr_w = e_utils.crop(tf.convert_to_tensor(hf['w'][idx]), pred_w[0])[tf.newaxis]
        
        # Relative error per frame
        rel_err[idx] = (loss_utils.calculate_relative_error(pred_u, pred_v, pred_w, hr_u, hr_v, hr_w, binary_mask))
        
        
        # Average speed per frame across all axis
        hr = tf.concat([hr_u, hr_v, hr_w], axis=0)
        squared = tf.map_fn(lambda x : tf.square(x), hr)
        speed = tf.math.sqrt(tf.reduce_sum(squared, axis=0))
        flow = tf.reduce_sum(speed, axis=[0,1,2]) / (tf.reduce_sum(binary_mask, axis=[1,2,3]) + 1)*100

        # Average speed per frame for each axis independetly.
        flow_uvw = tf.reduce_sum(hr, axis=[1,2,3]) / (tf.reduce_sum(binary_mask, axis=[1,2,3]) + 1)*100
        
        mean_speed[idx] = tf.concat([flow, flow_uvw], axis=0)
        if peak_flow < flow:
            peak_flow = flow
            peak_flow_idx = idx
        
    # Keep track of figure number or else graphs is plotted in same figure
    fig_nr = 1
    
    # fig_nr = e_utils.plot_relative_mean_error(rel_err, data_count, prediction_file, fig_nr)
    # fig_nr = e_utils.plot_mean_speed(mean_speed, data_count, prediction_file, fig_nr)
    fig_nr = e_utils.draw_reg_line(ground_truth_file, prediction_file, peak_flow_idx, tf.squeeze(binary_mask, axis=[0]), fig_nr)


    # Params to decide what to slice

    frame = 30
    lr_idx = 46
    vel_dir = 'u'
    slice_axis = 'x'

    # fig_nr = e_utils.generate_slice_comp([lr_file, ground_truth_file, prediction_file], frame, lr_idx, fig_nr, vel_dir, slice_axis)

    peak_flow_idx = 34
    # img3D = np.sqrt(u_sr[peak_flow_idx]**2 + v_sr[peak_flow_idx]**2 + w_sr[peak_flow_idx]**2)
    SR = u_sr[peak_flow_idx]
    # img3D = np.sqrt(u_hr[peak_flow_idx]**2 + v_hr[peak_flow_idx]**2 + w_hr[peak_flow_idx]**2)
    HR = u_hr[peak_flow_idx]

    absdiff = np.absolute(SR - HR)

    m = 1.1 #max(np.max(img3D), -np.min(img3D))
    #print(m)
    img3D = np.clip(SR/(2*m) + 0.5, 0.0, 1.0) # norm [0, 1]

    k = max(np.max(absdiff), -np.min(absdiff))
    print(k)
    k = 0.5
    img3Ddiff = np.clip(absdiff/(2*k) + 0.5, 0.0, 1.0) # norm [0, 1]

    # e_utils.generate_gif_volume(img3D, axis=2, save_as=f'{prediction_file[:-3]}_u', norm=False)

    # e_utils.generate_gif_volume(img3Ddiff, axis=2, save_as=f'{prediction_file[:-3]}_u_diff2', norm=False)

    mask_img = np_mask*0.5 + 0.5
    
    
    