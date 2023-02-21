import numpy as np
from Network.PatchHandler3D import PatchHandler3D

import matplotlib.pyplot as plt

def load_indexes(index_file):
    """
        Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index
    """
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes

if __name__ == "__main__":
    data_dir = '../../data/cerebro_data'
    
    # ---- Patch index files ----
    training_file = '{}/newtrain12.csv'.format(data_dir)
    validate_file = '{}/newval12.csv'.format(data_dir)
    benchmark_file = '{}/newbenchmark12.csv'.format(data_dir)

    # Hyperparameters optimisation variables
    batch_size = 1 # def 20
    mask_threshold = 0.6 # def 0.6

    patch_size = 12 # def 16
    res_increase = 2 # def 2

    # Load data file and indexes
    trainset = load_indexes(training_file)
    valset = load_indexes(validate_file)
    
    # ----------------- TensorFlow stuff -------------------
    # TRAIN dataset iterator
    z = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    trainset = z.initialize_dataset(trainset, shuffle=False, n_parallel=None)

    # VALIDATION iterator
    valdh = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    valset = valdh.initialize_dataset(valset, shuffle=False, n_parallel=None)

    n_bins = 100

    cov = 0
    i = 0

    u_dist = np.zeros(n_bins, dtype=np.int32)
    v_dist = np.zeros(n_bins, dtype=np.int32)
    w_dist = np.zeros(n_bins, dtype=np.int32)
    s_dist = np.zeros(n_bins, dtype=np.int32)

    for i, (data_pairs) in enumerate(trainset):
        u, v, w, u_mag, v_mag, w_mag, u_hr, v_hr, w_hr, venc, mask = data_pairs # LR-shape (12,12,12) , HR+mask-shape (24,24,24)

        coverage = np.sum(mask)/np.size(mask)
        #print(f'Coverage: {coverage}')
        cov += coverage

        m, n, o = mask[0].shape
        mask_lr = np.reshape(mask[0], (m//2, 2, n//2, 2, o//2, 2)).max((1, 3, 5))

        u_flow_lr = u[0,:,:,:,0]*mask_lr
        v_flow_lr = v[0,:,:,:,0]*mask_lr
        w_flow_lr = w[0,:,:,:,0]*mask_lr

        s_flow_lr = np.sqrt(u_flow_lr**2 + v_flow_lr**2 + w_flow_lr**2)

        hist, bin_edges = np.histogram(u_flow_lr, n_bins, range=(-1.0, 1.0), weights=mask_lr)
        u_dist += np.int32(hist) # * np.diff(bin_edges)
        hist, bin_edges = np.histogram(v_flow_lr, n_bins, range=(-1.0, 1.0), weights=mask_lr)
        v_dist += np.int32(hist) # * np.diff(bin_edges)
        hist, bin_edges = np.histogram(w_flow_lr, n_bins, range=(-1.0, 1.0), weights=mask_lr)
        w_dist += np.int32(hist) # * np.diff(bin_edges)

        hist, s_bin_edges = np.histogram(s_flow_lr, n_bins, range=(0.0, 1.0), weights=mask_lr)
        s_dist += np.int32(hist) # * np.diff(bin_edges)

        #print(u_dist)
        #print(v_dist)
        #print(w_dist)

        # plt.hist(np.array(u_flow_lr).flatten(), bins=20, weights=np.array(mask_lr).flatten(), density=True)
        # plt.title("u histogram")
        # plt.show()

        # if i > 500:
        #     break

    print(cov/(i+1))
    
    print(f'u distribution \n{u_dist}')
    print(f'v distribution \n{v_dist}')
    print(f'w distribution \n{w_dist}')
    print(f's distribution \n{s_dist}')
    #print(u_dist/np.sum(u_dist))

    fig, ax = plt.subplots(2, 3)
    fig.suptitle("u,v,w histograms")
    ax[0,0].hist(bin_edges[:-1], bin_edges, weights=u_dist, log=False)
    ax[1,0].hist(bin_edges[:-1], bin_edges, weights=u_dist, log=True)
    ax[0,1].hist(bin_edges[:-1], bin_edges, weights=v_dist, log=False)
    ax[1,1].hist(bin_edges[:-1], bin_edges, weights=v_dist, log=True)
    ax[0,2].hist(bin_edges[:-1], bin_edges, weights=w_dist, log=False)
    ax[1,2].hist(bin_edges[:-1], bin_edges, weights=w_dist, log=True)
    plt.show()
        
    fig, ax = plt.subplots(2, 1)
    fig.suptitle("Speed histogram")
    ax[0].hist(s_bin_edges[:-1], s_bin_edges, weights=s_dist, log=False)
    ax[1].hist(s_bin_edges[:-1], s_bin_edges, weights=s_dist, log=True)
    plt.show()

    