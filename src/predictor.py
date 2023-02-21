import tensorflow as tf
import numpy as np
import time
import os
from GANetwork.SR4DFlowGAN import SR4DFlowGAN
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.ImageDataset import ImageDataset


def prepare_network(patch_size, res_increase, low_resblock, hi_resblock):
    # Prepare input
    input_shape = (patch_size,patch_size,patch_size,1)
    output_shape = (res_increase*patch_size,res_increase*patch_size,res_increase*patch_size)

    u = tf.keras.layers.Input(shape=input_shape, name='u')
    v = tf.keras.layers.Input(shape=input_shape, name='v')
    w = tf.keras.layers.Input(shape=input_shape, name='w')

    # u_mag = tf.keras.layers.Input(shape=input_shape, name='u_mag')
    # v_mag = tf.keras.layers.Input(shape=input_shape, name='v_mag')
    # w_mag = tf.keras.layers.Input(shape=input_shape, name='w_mag')

    u_hr = tf.keras.layers.Input(shape=output_shape, name='u_hr')
    v_hr = tf.keras.layers.Input(shape=output_shape, name='v_hr')
    w_hr = tf.keras.layers.Input(shape=output_shape, name='w_hr')

    input_layer = [u,v,w]

    # network & output
    net = SR4DFlowGAN(patch_size, res_increase)
    generator = net.build_generator(u, v, w, low_resblock, hi_resblock)
    discriminator = net.build_disriminator(u_hr, v_hr, w_hr)
    model = net.build_network(u, v, w, generator, discriminator)

    return model

if __name__ == '__main__':
    data_dir = '../../data/cerebro_data'
    filename = 'patient1-075_LR.h5'
 
    output_dir = "../predictions"
    output_filename = 'patient1-075_PR.h5'
    
    model_path = "../models/4DFlowGAN_minitest_20230216-0059/4DFlowGAN_minitest-best.h5"
    # Params
    patch_size = 12 #24
    res_increase = 2
    batch_size = 8
    round_small_values = True

    # Network
    low_resblock=8
    hi_resblock=4

    # Setting up
    input_filepath = '{}/{}'.format(data_dir, filename)
    pgen = PatchGenerator(patch_size, res_increase)
    dataset = ImageDataset()

    # Check the number of rows in the file
    nr_rows = dataset.get_dataset_len(input_filepath)
    print(f"Number of rows in dataset: {nr_rows}")

    print(f"Loading 4DFlowNet: {res_increase}x upsample")
    # Load the network
    network = prepare_network(patch_size, res_increase, low_resblock, hi_resblock)
    network.load_weights(model_path)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # loop through all the rows in the input file
    for nrow in range(0, nr_rows):
        print("\n--------------------------")
        print(f"\nProcessed ({nrow+1}/{nr_rows}) - {time.ctime()}")
        # Load data file and indexes
        dataset.load_vectorfield(input_filepath, nrow)
        print(f"Original image shape: {dataset.u.shape}")
        
        velocities, magnitudes = pgen.patchify(dataset)
        data_size = len(velocities[0])
        print(f"Patchified. Nr of patches: {data_size} - {velocities[0].shape}")

        # Predict the patches
        #results = np.zeros((0,patch_size*res_increase, patch_size*res_increase, patch_size*res_increase, 3))
        results = np.zeros((0,patch_size*res_increase, patch_size*res_increase, patch_size*res_increase))
        start_time = time.time()

        for current_idx in range(0, data_size, batch_size):
            time_taken = time.time() - start_time
            print(f"\rProcessed {current_idx}/{data_size} Elapsed: {time_taken:.2f} secs.", end='\r')
            # Prepare the batch to predict
            patch_index = np.index_exp[current_idx:current_idx+batch_size]
            # sr_images = network.predict([velocities[0][patch_index],
            #                         velocities[1][patch_index],
            #                         velocities[2][patch_index],
            #                         magnitudes[0][patch_index],
            #                         magnitudes[1][patch_index],
            #                         magnitudes[2][patch_index]])
            sr_images = network.predict(velocities[0][patch_index])

            # results = np.append(results, sr_images, axis=0)
            results = np.append(results, sr_images[0], axis=0)
        # End of batch loop    
        time_taken = time.time() - start_time
        print(f"\rProcessed {data_size}/{data_size} Elapsed: {time_taken:.2f} secs.")

        #for i in range(0,3):
        for i in range(0,1):
            # v = pgen._patchup_with_overlap(results[:,:,:,:,i], pgen.nr_x, pgen.nr_y, pgen.nr_z)
            v = pgen._patchup_with_overlap(results[:,:,:,:], pgen.nr_x, pgen.nr_y, pgen.nr_z)
            
            # Denormalized
            v = v * dataset.venc 
            if round_small_values:
                # print(f"Zero out velocity component less than {dataset.velocity_per_px}")
                # remove small velocity values
                v[np.abs(v) < dataset.velocity_per_px] = 0
            
            v = np.expand_dims(v, axis=0) 
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', dataset.velocity_colnames[i], v, compression='gzip')

    if dataset.dx is not None:
        new_spacing = dataset.dx / res_increase
        new_spacing = np.expand_dims(new_spacing, axis=0) 
        prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', dataset.dx_colname, new_spacing, compression='gzip')

    print("Done!")