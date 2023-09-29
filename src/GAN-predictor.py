import tensorflow as tf
import numpy as np
import time
import os

from importlib import import_module

from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.ImageDataset import ImageDataset


def prepare_network(SR4DFlowGAN, patch_size, res_increase, low_resblock, hi_resblock):

    # network & output
    net = SR4DFlowGAN(patch_size, res_increase)
    generator = net.build_generator(low_resblock, hi_resblock, channel_nr=64)
    discriminator = net.build_disriminator(channel_nr=32)
    model = net.build_network(generator, discriminator)

    return model, generator

if __name__ == '__main__':

    # NOTE
    # All architecure parameters must match those used during training for the stored weights to be loaded successfully.
    # Relevant options are: patch size, res increase, low/hi resblocks, gen/disc channel count
    # If unsure about the values used, check the backup_source of the saved model.

    ### ---------------- SETTINGS ----------------

    GAN_module_name = "GANetwork"
    #GAN_module_name = "GANetworkU"
    #GAN_module_name = "GANetworkESR"

    data_dir = '../../data/cerebro_data'
    filename = 'patient3-postOp_LR.h5'
 
    output_dir = "../predictions/GAN"
    output_filename = 'patient3-postOp_SR.h5'

    model_path = "../models/4DFlowGAN_20230504-1327/4DFlowGAN-best.h5"

    # Params
    patch_size = 12 #24
    res_increase = 2
    batch_size = 8
    round_small_values = True

    # Network
    low_resblock=4
    hi_resblock=2

    ### ------------------------------------------------


    # Setting up
    input_filepath = '{}/{}'.format(data_dir, filename)
    pgen = PatchGenerator(patch_size, res_increase)
    dataset = ImageDataset(use_mag=False)

    # Check the number of rows in the file
    nr_rows = dataset.get_dataset_len(input_filepath)
    print(f"Number of rows in dataset: {nr_rows}")

    # Dynamic import of GAN module's architecture 
    SR4DFlowGAN = import_module(GAN_module_name + '.SR4DFlowGAN', 'src').SR4DFlowGAN

    print(f"Loading {GAN_module_name}: {res_increase}x upsample")
    # Load the network
    network, generator = prepare_network(SR4DFlowGAN, patch_size, res_increase, low_resblock, hi_resblock)
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
        
        velocities, magnitudes = pgen.patchify(dataset, use_mag=False)
        data_size = len(velocities[0])
        print(f"Patchified. Nr of patches: {data_size} - {velocities[0].shape}")

        # Predict the patches
        results = np.zeros((0,patch_size*res_increase, patch_size*res_increase, patch_size*res_increase, 3))
        start_time = time.time()

        for current_idx in range(0, data_size, batch_size):
            time_taken = time.time() - start_time
            print(f"\rProcessed {current_idx}/{data_size} Elapsed: {time_taken:.2f} secs.", end='\r')
            # Prepare the batch to predict
            patch_index = np.index_exp[current_idx:current_idx+batch_size]
            input_data = np.concatenate([velocities[0][patch_index],
                                    velocities[1][patch_index],
                                    velocities[2][patch_index]], axis=-1)
            sr_images = generator.predict(input_data)

            results = np.append(results, sr_images, axis=0)
        # End of batch loop    
        time_taken = time.time() - start_time
        print(f"\rProcessed {data_size}/{data_size} Elapsed: {time_taken:.2f} secs.")

        for i in range(0,3):
            v = pgen._patchup_with_overlap(results[:,:,:,:,i], pgen.nr_x, pgen.nr_y, pgen.nr_z)
            
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