"""
4DFlowGAN: Super Resolution GAN
Author: David Bardvall
Date:   13/02/2023
"""
import pickle
import tensorflow as tf
import numpy as np
import datetime
import time
import shutil
import os
from .SR4DFlowGAN import SR4DFlowGAN
from Network import utility, h5util, loss_utils

cross_kernel = tf.constant(np.array([[[0,0,0],
                                      [0,1,0],
                                      [0,0,0]],
                                     [[0,1,0],
                                      [1,1,1],
                                      [0,1,0]],
                                     [[0,0,0],
                                      [0,1,0],
                                      [0,0,0]]]), dtype=tf.float32)

ball_kernel =  tf.constant(np.array([[[0,1,0],
                                      [1,1,1],
                                      [0,1,0]],
                                     [[1,1,1],
                                      [1,1,1],
                                      [1,1,1]],
                                     [[0,1,0],
                                      [1,1,1],
                                      [0,1,0]]]), dtype=tf.float32)

def expand_mask(mask):
    padded = tf.pad(mask, [[0,0],[1,1],[1,1],[1,1]], 'SYMMETRIC')
    conved = tf.nn.conv3d(padded[..., tf.newaxis], ball_kernel[..., tf.newaxis, tf.newaxis], [1,1,1,1,1], 'VALID')
    new_mask = tf.cast(tf.greater(conved[..., 0], tf.constant(0.5)), dtype=tf.float32)
    return new_mask

class TrainerController:
    # constructor
    def __init__(self, patch_size, res_increase, initial_learning_rate=1e-4, quicksave_enable=True, network_name='4DFlowGAN', low_resblock=8, hi_resblock=4):
        """
            TrainerController constructor
            Setup all the placeholders, network graph, loss functions and optimizer here.
        """
        self.div_weight = 0 # Weighting for divergence loss
        self.non_fluid_weight = 1.0 # Weigthing for non fluid region
        self.boundary_weight = 1.0
        self.disc_loss_weight = 0.00001
        self.discL2weight = 500

        self.mask_for_disc = True # Apply mask to SR field before being passed to the discriminator

        # General param
        self.res_increase = res_increase
        
        # Training params
        self.QUICKSAVE_ENABLED = quicksave_enable
        
        # Network
        self.network_name = network_name

        net = SR4DFlowGAN(patch_size, res_increase)
        self.generator = net.build_generator(low_resblock, hi_resblock, channel_nr=64)
        self.discriminator = net.build_disriminator(channel_nr=32)
        self.model = net.build_network(self.generator, self.discriminator)

        # ===== Metrics =====
        self.loss_metrics = dict([
            ('train_gen_loss', tf.keras.metrics.Mean(name='train_gen_loss')),
            ('val_gen_loss', tf.keras.metrics.Mean(name='val_gen_loss')),
            ('train_accuracy', tf.keras.metrics.Mean(name='train_accuracy')),
            ('val_accuracy', tf.keras.metrics.Mean(name='val_accuracy')),
            ('train_gen_mse', tf.keras.metrics.Mean(name='train_gen_mse')),
            ('val_gen_mse', tf.keras.metrics.Mean(name='val_gen_mse')),
            ('train_gen_adv_loss', tf.keras.metrics.Mean(name='train_gen_adv_loss')),
            ('val_gen_adv_loss', tf.keras.metrics.Mean(name='val_gen_adv_loss')),
            # ('train_div', tf.keras.metrics.Mean(name='train_div')),
            # ('val_div', tf.keras.metrics.Mean(name='val_div')),
            ('train_disc_loss', tf.keras.metrics.Mean(name='train_disc_loss')),
            ('val_disc_loss', tf.keras.metrics.Mean(name='val_disc_loss')),
            ('train_disc_adv_loss', tf.keras.metrics.Mean(name='train_disc_adv_loss')),
            ('val_disc_adv_loss', tf.keras.metrics.Mean(name='val_disc_adv_loss')),

            ('l2_gen_loss', tf.keras.metrics.Mean(name='l2_gen_loss')),
            ('l2_disc_loss', tf.keras.metrics.Mean(name='l2_disc_loss')),
        ])
        self.accuracy_metric = 'val_gen_mse'
        
        #print(f"Divergence loss2 * {self.div_weight}")
        print(f"Accuracy metric: {self.accuracy_metric}")

        # learning rate and training optimizer
        self.learning_rate = initial_learning_rate
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Compile model so we can save the optimizer weights
        # self.model.compile(loss=self.loss_function, optimizer=self.optimizer)

    def save_latest_model(self, epoch):
        if epoch > 0 and epoch % 10 == 0:
            self.model.save(f'{self.model_path}-latest.h5')
            message = f'Saving current model - {time.ctime()}\n'
            print(message)

    def loss_function(self, y_true, y_pred, mask):
        """
            Calculate Total Loss function
            Loss = MSE + weight * div_loss2
        """
        u,v,w = y_true[...,0],y_true[...,1], y_true[...,2]
        u_pred,v_pred,w_pred = y_pred[...,0],y_pred[...,1], y_pred[...,2]

        mse = self.calculate_mse(u,v,w, u_pred,v_pred,w_pred)

        # if mask is not None:
        # === Separate mse ===
        non_fluid_mask = tf.less(mask, tf.constant(0.5))
        non_fluid_mask = tf.cast(non_fluid_mask, dtype=tf.float32)

        epsilon = 1 # minimum 1 pixel

        if self.boundary_weight != 1:
            padded = tf.pad(mask, [[0,0],[1,1],[1,1],[1,1]], 'SYMMETRIC')
            conved = tf.nn.conv3d(padded[..., tf.newaxis], cross_kernel[..., tf.newaxis, tf.newaxis], [1,1,1,1,1], 'VALID')
            core_mask = tf.cast(tf.greater(conved[..., 0], tf.constant(6.5)), dtype=tf.float32)
            boundary_mask = mask - core_mask
            mask = core_mask + boundary_mask * self.boundary_weight

        fluid_mse = mse * mask
        fluid_mse = tf.reduce_sum(fluid_mse, axis=[1,2,3]) / (tf.reduce_sum(mask, axis=[1,2,3]) + epsilon)

        non_fluid_mse = mse * non_fluid_mask
        non_fluid_mse = tf.reduce_sum(non_fluid_mse, axis=[1,2,3]) / (tf.reduce_sum(non_fluid_mask, axis=[1,2,3]) + epsilon)

        mse = fluid_mse + non_fluid_mse * self.non_fluid_weight

        # divergence
        
        # divergence_loss = loss_utils.calculate_divergence_loss2(u,v,w, u_pred,v_pred,w_pred)
        # divergence_loss = self.div_weight * divergence_loss

        # fluid_divloss = divergence_loss * mask
        # fluid_divloss = tf.reduce_sum(fluid_divloss, axis=[1,2,3]) / (tf.reduce_sum(mask, axis=[1,2,3]) + epsilon)

        # non_fluid_divloss = divergence_loss * non_fluid_mask
        # non_fluid_divloss = tf.reduce_sum(non_fluid_divloss, axis=[1,2,3]) / (tf.reduce_sum(non_fluid_mask, axis=[1,2,3]) + epsilon)

        # divergence_loss = fluid_divloss + non_fluid_divloss
        divergence_loss = 0

        # standard without masking
        total_loss = tf.reduce_mean(mse + divergence_loss)

        # return all losses for logging
        return  total_loss, mse, divergence_loss

    def calculate_regularizer_loss(self, net=None):
        """
            https://stackoverflow.com/questions/62440162/how-do-i-take-l1-and-l2-regularizers-into-account-in-tensorflow-custom-training
        """
        loss = 0
        if net is None:
            net = self.model
        for l in net.layers:
            # if hasattr(l,'layers') and l.layers: # the layer itself is a model
            #     loss+=add_model_loss(l)
            if hasattr(l,'kernel_regularizer') and l.kernel_regularizer:
                loss+=l.kernel_regularizer(l.kernel)
            if hasattr(l,'bias_regularizer') and l.bias_regularizer:
                loss+=l.bias_regularizer(l.bias)
        return loss

    def accuracy_function(self, y_true, y_pred, mask):
        """
            Calculate relative speed error
        """
        u,v,w = y_true[...,0],y_true[...,1], y_true[...,2]
        u_pred,v_pred,w_pred = y_pred[...,0],y_pred[...,1], y_pred[...,2]

        return loss_utils.calculate_relative_error(u_pred, v_pred, w_pred, u, v, w, mask)

    def calculate_mse(self, u, v, w, u_pred, v_pred, w_pred):
        """
            Calculate Speed magnitude error
        """
        return (u_pred - u) ** 2 +  (v_pred - v) ** 2 + (w_pred - w) ** 2

    def calculate_bce(self, target, pred):
        """
            Calculate Binary Cross Entropy
        """
        return tf.keras.losses.BinaryCrossentropy()(target, pred)

    def init_model_dir(self):
        """
            Create model directory to save the weights with a [network_name]_[datetime] format
            Also prepare logfile and tensorboard summary within the directory.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.unique_model_name = f'{self.network_name}_{timestamp}'

        self.model_dir = f"../models/{self.unique_model_name}"
        # Do not use .ckpt on the model_path
        self.model_path = f"{self.model_dir}/{self.network_name}"

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        # summary - Tensorboard stuff
        self._prepare_logfile_and_summary()
    
    def _prepare_logfile_and_summary(self):
        """
            Prepare csv logfile to keep track of the loss and Tensorboard summaries
        """
        # summary - Tensorboard stuff
        self.train_writer = tf.summary.create_file_writer(self.model_dir+'/tensorboard/train')
        self.val_writer = tf.summary.create_file_writer(self.model_dir+'/tensorboard/validate')

        # Prepare log file
        self.logfile = self.model_dir + '/loss.csv'

        utility.log_to_file(self.logfile, f'Network: {self.network_name}\n')
        utility.log_to_file(self.logfile, f'Initial learning rate: {self.learning_rate}\n')
        utility.log_to_file(self.logfile, f'Accuracy metric: {self.accuracy_metric}\n')
        utility.log_to_file(self.logfile, f'Divergence weight: {self.div_weight}\n')

        # Header
        stat_names = ','.join(self.loss_metrics.keys()) # train and val stat names
        utility.log_to_file(self.logfile, f'epoch, {stat_names}, learning rate, elapsed (sec), best_model, benchmark_err, benchmark_rel_err, benchmark_mse, benchmark_divloss\n')

        print("Copying source code to model directory...")
        # Copy all the source file to the model dir for backup
        directory_to_backup = [".", "GANetworkX"]
        for directory in directory_to_backup:
            files = os.listdir(directory)
            for fname in files:
                if fname.endswith(".py") or fname.endswith(".ipynb"):
                    dest_fpath = os.path.join(self.model_dir,"backup_source",directory, fname)
                    os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)

                    shutil.copy2(f"{directory}/{fname}", dest_fpath)

      
    @tf.function
    def train_step(self, data_pairs):
        u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
        x_target = tf.concat((u_hr, v_hr, w_hr), axis=-1)
        train_mask = expand_mask(mask)

        with tf.GradientTape() as gen_tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            input_data = tf.concat([u,v,w], axis=-1)
            x_pred = self.generator(input_data, training=True)
            x_pred_p = x_pred * tf.expand_dims(train_mask, -1) if self.mask_for_disc else x_pred

            self.discriminator.trainable = True
            with tf.GradientTape() as disc_tape:
                real_y_pred = self.discriminator(x_target, training=True)
                fake_y_pred = self.discriminator(x_pred_p, training=True)
                real_y_target = np.ones(fake_y_pred.shape)
                fake_y_target = np.zeros(fake_y_pred.shape)

                disc_loss = self.disc_calculate_and_update_metrics(fake_y_target, fake_y_pred, real_y_target, real_y_pred, 'train')

            disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
            self.discriminator.trainable = False

            gen_loss = self.gen_calculate_and_update_metrics(x_target, x_pred, real_y_target, fake_y_pred, train_mask, mask, 'train')

        # Get the gradients
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        # Update the weights
        self.optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        

    @tf.function
    def test_step(self, data_pairs):
        u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
        x_target = tf.concat((u_hr, v_hr, w_hr), axis=-1)
        train_mask = expand_mask(mask)
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        input_data = tf.concat([u,v,w], axis=-1)
        x_pred = self.generator(input_data, training=False)
        x_pred_p = x_pred * tf.expand_dims(train_mask, -1) if self.mask_for_disc else x_pred
        fake_y_pred = self.discriminator(x_pred_p, training=False)

        real_y_target = np.ones(fake_y_pred.shape)
        fake_y_target = np.zeros(fake_y_pred.shape)

        real_y_pred = self.discriminator(x_target, training=False)

        disc_loss = self.disc_calculate_and_update_metrics(fake_y_target, fake_y_pred, real_y_target, real_y_pred, 'val')
        self.loss_metrics['val_disc_loss'].update_state(disc_loss)
        
        self.gen_calculate_and_update_metrics(x_target, x_pred, real_y_target, fake_y_pred, train_mask, mask, 'val')
       
        return x_pred

    def gen_calculate_and_update_metrics(self, hires, predictions, real_y_target, fake_y_pred, train_mask, mask, metric_set):
        loss, mse, divloss = self.loss_function(hires, predictions, train_mask)
        rel_error = self.accuracy_function(hires, predictions, mask)
        
        if metric_set == 'train':
            l2_reg_loss = self.calculate_regularizer_loss(self.generator)
            self.loss_metrics[f'l2_gen_loss'].update_state(l2_reg_loss)

            loss += l2_reg_loss

        adv_loss = self.disc_loss_weight * self.calculate_bce(real_y_target, fake_y_pred)
        self.loss_metrics[f'{metric_set}_gen_adv_loss'].update_state(adv_loss)

        loss += adv_loss

        # Update the loss and accuracy
        self.loss_metrics[f'{metric_set}_gen_loss'].update_state(loss)

        self.loss_metrics[f'{metric_set}_gen_mse'].update_state(mse)
        # self.loss_metrics[f'{metric_set}_div'].update_state(divloss)
        self.loss_metrics[f'{metric_set}_accuracy'].update_state(rel_error)
        return loss
    
    def disc_calculate_and_update_metrics(self, fake_y_target, fake_y_pred, real_y_target, real_y_pred, metric_set):
        real_disc_loss = self.calculate_bce(real_y_target, real_y_pred)
        fake_disc_loss = self.calculate_bce(fake_y_target, fake_y_pred)
        adv_loss = 0.5 * (real_disc_loss + fake_disc_loss)
        self.loss_metrics[f'{metric_set}_disc_adv_loss'].update_state(adv_loss)

        disc_loss = adv_loss
        
        if metric_set == 'train':
            l2_reg_loss = self.calculate_regularizer_loss(self.discriminator) * self.discL2weight
            self.loss_metrics[f'l2_disc_loss'].update_state(l2_reg_loss)
            disc_loss += l2_reg_loss

        self.loss_metrics[f'{metric_set}_disc_loss'].update_state(disc_loss)

        return disc_loss

    def reset_metrics(self):
        for key in self.loss_metrics.keys():
            self.loss_metrics[key].reset_states()

    def train_network(self, trainset, valset, n_epoch, testset=None):
        """
            Main training function. Receives trainining and validation TF dataset.
        """
        # ----- Run the training -----
        print("==================== TRAINING =================")
        print(f'Learning rate {self.optimizer.lr.numpy():.7f}')
        print(f"Start training at {time.ctime()} - {self.unique_model_name}\n")
        start_time = time.time()
        
        # Setup acc and data count
        previous_loss = np.inf
        total_batch_train = tf.data.experimental.cardinality(trainset).numpy()
        total_batch_val = tf.data.experimental.cardinality(valset).numpy()

        for epoch in range(n_epoch):
            # ------------------------------- Training -------------------------------
            # self.adjust_learning_rate(epoch)

            # Reset the metrics at the start of the next epoch
            self.reset_metrics()

            start_loop = time.time()
            # --- Training ---
            for i, (data_pairs) in enumerate(trainset):
                # Train the network
                self.train_step(data_pairs)
                message = f"Epoch {epoch+1} Train batch {i+1}/{total_batch_train} | loss: {self.loss_metrics['train_gen_loss'].result():.5f} ({self.loss_metrics['train_accuracy'].result():.1f} %) - {time.time()-start_loop:.1f} secs"
                print(f"\r{message}", end='')

            # --- Validation ---
            for i, (data_pairs) in enumerate(valset):
                self.test_step(data_pairs)
                message = f"Epoch {epoch+1} Validation batch {i+1}/{total_batch_val} | loss: {self.loss_metrics['val_gen_loss'].result():.5f} ({self.loss_metrics['val_accuracy'].result():.1f} %) - {time.time()-start_loop:.1f} secs"
                print(f"\r{message}", end='')

            # --- Epoch logging ---
            message = f"\rEpoch {epoch+1} Train loss: {self.loss_metrics['train_gen_loss'].result():.5f} ({self.loss_metrics['train_accuracy'].result():.1f} %), Val loss: {self.loss_metrics['val_gen_loss'].result():.5f} ({self.loss_metrics['val_accuracy'].result():.1f} %) - {time.time()-start_loop:.1f} secs"
            
            loss_values = []
            # Get the loss values from the loss_metrics dict
            for key, value in self.loss_metrics.items():
                # TODO: handle formatting here
                loss_values.append(f'{value.result():.5f}')
            loss_str = ','.join(loss_values)
            log_line = f"{epoch+1},{loss_str},{self.optimizer.lr.numpy():.6f},{time.time()-start_loop:.1f}"
            

            self._update_summary_logging(epoch)

            # --- Save criteria ---
            if self.loss_metrics[self.accuracy_metric].result() < previous_loss or epoch == n_epoch-5:
                self.save_best_model()
                
                # Update best acc
                previous_loss = self.loss_metrics[self.accuracy_metric].result()
                
                # logging
                message  += ' **' # Mark as saved
                log_line += ',**'

                # Benchmarking
                if self.QUICKSAVE_ENABLED and testset is not None:
                    quick_loss, quick_accuracy, quick_mse, quick_div = self.quicksave(testset, epoch+1)
                    quick_loss, quick_accuracy, quick_mse, quick_div = np.mean(quick_loss), np.mean(quick_accuracy), np.mean(quick_mse), np.mean(quick_div)

                    message  += f' Benchmark loss: {quick_loss:.5f} ({quick_accuracy:.1f} %)'
                    log_line += f', {quick_loss:.7f}, {quick_accuracy:.2f}%, {quick_mse:.7f}, {quick_div:.7f}'
            # Logging
            print(message)
            utility.log_to_file(self.logfile, log_line+"\n")
            # /END of epoch loop

        # End
        hrs, mins, secs = utility.calculate_time_elapsed(start_time)
        message =  f"\nTraining {self.network_name} completed! - name: {self.unique_model_name}"
        message += f"\nTotal training time: {hrs} hrs {mins} mins {secs} secs."
        message += f"\nFinished at {time.ctime()}"
        message += f"\n==================== END TRAINING ================="
        utility.log_to_file(self.logfile, message)
        print(message)
        
        # Finish!
        
    def save_best_model(self):
        """
            Save model weights and also optmizer weights to enable restore model
            to continue training

            Based on:
            https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
        """
        # Save model weights.
        self.model.save(f'{self.model_path}-best.h5')
        
        # Save optimizer weights.
        symbolic_weights = getattr(self.optimizer, 'weights', None)
        if symbolic_weights is None:
            print("WARN --- MANUALFIX USED")
        if symbolic_weights:
            weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
            with open(f'{self.model_dir}/optimizer.pkl', 'wb') as f:
                pickle.dump(weight_values, f)

    def restore_model(self, old_model_dir, old_model_file):
        """
            Restore model weights and optimizer weights for uncompiled model
            Based on: https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state

            For an uncompiled model, we cannot just set the optmizer weights directly because they are zero.
            We need to at least do an apply_gradients once and then set the optimizer weights.
        """
        # Set the path for the weights and optimizer
        model_weights_path = f"{old_model_dir}/{old_model_file}"
        opt_path   = f"{old_model_dir}/optimizer.pkl"

        # Load the optimizer weights
        with open(opt_path, 'rb') as f:
            opt_weights = pickle.load(f)
        
        # Get the model's trainable weights
        grad_vars_gen = self.generator.trainable_weights
        grad_vars_disc = self.discriminator.trainable_weights
        # This need not be model.trainable_weights; it must be a correctly-ordered list of 
        # grad_vars corresponding to how you usually call the optimizer.
        zero_grads_gen = [tf.zeros_like(w) for w in grad_vars_gen]
        zero_grads_disc = [tf.zeros_like(w) for w in grad_vars_disc]

        # Apply gradients which don't do nothing with Adam
        self.optimizer.apply_gradients(zip(zero_grads_disc, grad_vars_disc))
        self.optimizer.apply_gradients(zip(zero_grads_gen, grad_vars_gen))

        # Set the weights of the optimizer
        self.optimizer.set_weights(opt_weights)

        # NOW set the trainable weights of the model
        self.model.load_weights(model_weights_path)

    def _update_summary_logging(self, epoch):
        """
            Tf.summary for epoch level loss
        """
        # Filter out the train and val metrics
        train_metrics = {k.replace('train_',''): v for k, v in self.loss_metrics.items() if k.startswith('train_')}
        val_metrics = {k.replace('val_',''): v for k, v in self.loss_metrics.items() if k.startswith('val_')}
        
        # Summary writer
        with self.train_writer.as_default():
            #tf.summary.scalar(f"{self.network_name}/learning_rate", self.optimizer.lr, step=epoch)
            tf.summary.scalar(f"{self.network_name}/l2_gen_loss", self.loss_metrics['l2_gen_loss'].result(), step=epoch)
            tf.summary.scalar(f"{self.network_name}/l2_disc_loss", self.loss_metrics['l2_disc_loss'].result(), step=epoch)
            for key in train_metrics.keys():
                tf.summary.scalar(f"{self.network_name}/{key}",  train_metrics[key].result(), step=epoch)         
        
        with self.val_writer.as_default():
            for key in val_metrics.keys():
                tf.summary.scalar(f"{self.network_name}/{key}",  val_metrics[key].result(), step=epoch)
   
        
    def quicksave(self, testset, epoch_nr):
        """
            Predict a batch of data from the benchmark testset.
            This is saved under the model directory with the name quicksave_[network_name].h5
            Quicksave is done everytime the best model is saved.
        """
        for i, (data_pairs) in enumerate(testset):
            u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
            hires = tf.concat((u_hr, v_hr, w_hr), axis=-1)
            input_data = [u,v,w]

            preds = self.generator.predict(input_data)

            loss_val, mse, divloss = self.loss_function(hires, preds, mask)
            rel_loss = self.accuracy_function(hires, preds, mask)
            # Do only 1 batch
            break

        quicksave_filename = f"quicksave_{self.network_name}.h5"
        h5util.save_predictions(self.model_dir, quicksave_filename, "epoch", np.asarray([epoch_nr]), compression='gzip')

        preds = np.expand_dims(preds, 0) # Expand dim to [epoch_nr, batch, ....]
        
        h5util.save_predictions(self.model_dir, quicksave_filename, "u", preds[...,0], compression='gzip')
        h5util.save_predictions(self.model_dir, quicksave_filename, "v", preds[...,1], compression='gzip')
        h5util.save_predictions(self.model_dir, quicksave_filename, "w", preds[...,2], compression='gzip')

        if epoch_nr == 1:
            # Save the actual data only for the first epoch
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_u", u, compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_v", v, compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_w", w, compression='gzip')

            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_u", np.squeeze(u_hr, -1), compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_v", np.squeeze(v_hr, -1), compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_w", np.squeeze(w_hr, -1), compression='gzip')
            
            h5util.save_predictions(self.model_dir, quicksave_filename, "venc", venc, compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "mask", mask, compression='gzip')
        
        return loss_val, rel_loss, mse, divloss