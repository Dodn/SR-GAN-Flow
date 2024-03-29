import tensorflow as tf

reg_l2 = tf.keras.regularizers.l2(5e-7)

class SR4DFlowGAN():
    def __init__(self, patch_size, res_increase):
        self.patch_size = patch_size
        self.res_increase = res_increase

    def build_network(self, generator=None, discriminator=None):
        
        input_shape = (self.patch_size,self.patch_size,self.patch_size,3)

        inputs = tf.keras.layers.Input(shape=input_shape, name='uvw')

        if generator is None:
            generator = self.build_generator()

        x = generator(inputs)

        if discriminator is None:
            discriminator = self.build_disriminator()

        y = discriminator([x, inputs])

        model = tf.keras.Model(inputs=inputs, outputs = [x, y], name='GAN')

        return model


    def build_generator(self, low_resblock=8, hi_resblock=4, channel_nr=64):

        input_shape = (self.patch_size,self.patch_size,self.patch_size,3)

        inputs = tf.keras.layers.Input(shape=input_shape, name='uvw')
    
        phase = conv3d(inputs, 3, channel_nr//2, 'SYMMETRIC', 'relu')

        phase = conv3d(phase, 3, channel_nr, 'SYMMETRIC', 'relu')
        
        # res blocks
        rb = phase
        for _ in range(low_resblock):
            rb = resnet_block(rb, channel_nr, pad='SYMMETRIC')

        rb = upsample3d(rb, self.res_increase)

        # refinement in HR
        for _ in range(hi_resblock):
            rb = resnet_block(rb, channel_nr, pad='SYMMETRIC')

        # 3 separate path version
        u_path = conv3d(rb, 3, channel_nr//2, 'SYMMETRIC', 'relu')
        u_path = conv3d(u_path, 3, 1, 'SYMMETRIC', None)

        v_path = conv3d(rb, 3, channel_nr//2, 'SYMMETRIC', 'relu')
        v_path = conv3d(v_path, 3, 1, 'SYMMETRIC', None)

        w_path = conv3d(rb, 3, channel_nr//2, 'SYMMETRIC', 'relu')
        w_path = conv3d(w_path, 3, 1, 'SYMMETRIC', None)
        

        b_out = tf.keras.layers.concatenate([u_path, v_path, w_path])

        model = tf.keras.Model(inputs=inputs, outputs = b_out, name='Generator')

        return model


    def build_disriminator(self, channel_nr=64):
        hr_dim = self.patch_size*self.res_increase
        lr_dim = self.patch_size

        input_shape = (hr_dim,hr_dim,hr_dim,3)
        lr_input_shape = (lr_dim,lr_dim,lr_dim,3)

        inputs = tf.keras.layers.Input(shape=input_shape, name='uvw_hr')
        lr_inputs = tf.keras.layers.Input(shape=lr_input_shape, name='uvw_lr')

        feat = conv3d(inputs, 3, channel_nr, 'SYMMETRIC')

        cur_dim = hr_dim

        feat = disc_block(feat, channel_nr, 2, pad='SYMMETRIC')
        channel_nr = min(channel_nr * 2, 128)
        feat = disc_block(feat, channel_nr, 1, pad='SYMMETRIC')

        feat = tf.keras.layers.concatenate([feat, lr_inputs])
        feat = disc_block(feat, channel_nr, 1, pad='SYMMETRIC')
        
        feat = disc_block(feat, channel_nr, 2, pad='SYMMETRIC')
        channel_nr = min(channel_nr * 2, 128)
        feat = disc_block(feat, channel_nr, 1, pad='SYMMETRIC')

        feat = disc_block(feat, channel_nr, 2, pad='SYMMETRIC')
        channel_nr = min(channel_nr * 2, 128)
        feat = disc_block(feat, channel_nr, 1, pad='SYMMETRIC')

        feat = tf.keras.layers.Flatten()(feat)
        feat = tf.keras.layers.Dense(64, kernel_regularizer=reg_l2, bias_regularizer=reg_l2)(feat)
        feat = tf.keras.layers.LeakyReLU(alpha = 0.2)(feat)
        y = tf.keras.layers.Dense(1, kernel_regularizer=reg_l2, bias_regularizer=reg_l2)(feat)
        y = tf.keras.layers.Activation('sigmoid')(y)

        epsilon = 0.0001
        y = epsilon + y * (1-2*epsilon)

        model = tf.keras.Model(inputs=[inputs, lr_inputs], outputs = y, name='Discriminator')

        return model


def upsample3d(input_tensor, res_increase):
    """
        Resize the image by linearly interpolating the input
        using TF '``'resize_bilinear' function.

        :param input_tensor: 2D/3D image tensor, with shape:
            'batch, X, Y, Z, Channels'
        :return: interpolated volume

        Original source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/linear_resize.html
    """
    
    # We need this option for the bilinear resize to prevent shifting bug
    align = True 

    b_size, x_size, y_size, z_size, c_size = input_tensor.shape

    x_size_new, y_size_new, z_size_new = x_size * res_increase, y_size * res_increase, z_size * res_increase

    if res_increase == 1:
        # already in the target shape
        return input_tensor

    # resize y-z
    squeeze_b_x = tf.reshape(input_tensor, [-1, y_size, z_size, c_size], name='reshape_bx')
    #resize_b_x = tf.compat.v1.image.resize_bilinear(squeeze_b_x, [y_size_new, z_size_new], align_corners=align)
    resize_b_x = tf.image.resize(squeeze_b_x, [y_size_new, z_size_new])
    resume_b_x = tf.reshape(resize_b_x, [-1, x_size, y_size_new, z_size_new, c_size], name='resume_bx')

    # Reorient
    reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
    
    #   squeeze and 2d resize
    squeeze_b_z = tf.reshape(reoriented, [-1, y_size_new, x_size, c_size], name='reshape_bz')
    #resize_b_z = tf.compat.v1.image.resize_bilinear(squeeze_b_z, [y_size_new, x_size_new], align_corners=align)
    resize_b_z = tf.image.resize(squeeze_b_z, [y_size_new, x_size_new])
    resume_b_z = tf.reshape(resize_b_z, [-1, z_size_new, y_size_new, x_size_new, c_size], name='resume_bz')
    
    output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
    return output_tensor


def conv3d(x, kernel_size, filters, padding='SYMMETRIC', activation=None, initialization=None, use_bias=True, strides=1):
    """
        Based on: https://github.com/gitlimlab/CycleGAN-Tensorflow/blob/master/ops.py
        For tf padding, refer to: https://www.tensorflow.org/api_docs/python/tf/pad
    """

    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = (kernel_size - 1) // 2
        x = tf.pad(x, [[0,0],[p,p],[p,p], [p,p],[0,0]], padding)
        x = tf.keras.layers.Conv3D(filters, kernel_size, strides = strides, activation=activation, kernel_initializer=initialization, use_bias=use_bias, kernel_regularizer=reg_l2, bias_regularizer=reg_l2 if use_bias else None)(x)
    else:
        assert padding in ['SAME', 'VALID']
        x = tf.keras.layers.Conv3D(filters, kernel_size, strides = strides, activation=activation, kernel_initializer=initialization, use_bias=use_bias, kernel_regularizer=reg_l2, bias_regularizer=reg_l2 if use_bias else None)(x)
    return x
    

def resnet_block(x, channel_nr=64, scale = 1, pad='SAME'):
    tmp = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

    tmp = conv3d(tmp, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)

    tmp = x + tmp * scale
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

    return tmp


def conv2d(x, kernel_size, filters, padding='SYMMETRIC', activation=None, initialization=None, use_bias=True, strides=1):
    """
        Based on: https://github.com/gitlimlab/CycleGAN-Tensorflow/blob/master/ops.py
        For tf padding, refer to: https://www.tensorflow.org/api_docs/python/tf/pad
    """

    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = (kernel_size - 1) // 2
        x = tf.pad(x, [[0,0],[p,p],[p,p],[0,0]], padding)
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, activation=activation, kernel_initializer=initialization, use_bias=use_bias, kernel_regularizer=reg_l2)(x)
    else:
        assert padding in ['SAME', 'VALID']
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides = strides, activation=activation, kernel_initializer=initialization, use_bias=use_bias, kernel_regularizer=reg_l2)(x)
    return x
    

def cnn_block(x, channel_nr=64, pad='SAME'):
    tmp = conv2d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

    tmp = conv2d(tmp, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)

    tmp = x + tmp
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

    return tmp

def disc_block(x, channel_nr=64, strides=1, pad='SAME'):
    tmp = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, strides=strides, activation=None, use_bias=True, initialization=None)
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

    return tmp
