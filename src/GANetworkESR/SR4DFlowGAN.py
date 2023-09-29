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

        y = discriminator(x)

        model = tf.keras.Model(inputs=inputs, outputs = [x, y], name='GAN')

        return model


    def build_generator(self, low_resblock=4, hi_resblock=2, channel_nr=32):

        input_shape = (self.patch_size,self.patch_size,self.patch_size,3)

        inputs = tf.keras.layers.Input(shape=input_shape, name='uvw')
    
        phase = conv3d(inputs, 3, channel_nr//2, 'SYMMETRIC', 'relu')

        phase = conv3d(phase, 3, channel_nr, 'SYMMETRIC', 'relu')
        
        # res blocks
        rb = phase
        for _ in range(low_resblock):
            rb = RRD_block(rb, channel_nr, pad='SYMMETRIC')

        rb = upsample3d(phase + rb * 0.2, self.res_increase)

        pre = rb

        # refinement in HR
        for _ in range(hi_resblock):
            rb = RRD_block(rb, channel_nr, pad='SYMMETRIC')

        rb = pre + rb * 0.2

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

        input_shape = (hr_dim,hr_dim,hr_dim,3)

        inputs = tf.keras.layers.Input(shape=input_shape, name='uvw_hr')

        feat = conv3d(inputs, 3, channel_nr, 'SYMMETRIC')

        cur_dim = hr_dim

        while cur_dim > 3:
            cur_dim = cur_dim / 2
            feat = disc_block(feat, channel_nr, 2, pad='SYMMETRIC')
            channel_nr = min(channel_nr * 2, 128)
            feat = disc_block(feat, channel_nr, 1, pad='SYMMETRIC')

        feat = tf.keras.layers.Flatten()(feat)
        feat = tf.keras.layers.Dense(64, kernel_regularizer=reg_l2, bias_regularizer=reg_l2)(feat)
        feat = tf.keras.layers.LeakyReLU(alpha = 0.2)(feat)
        y = tf.keras.layers.Dense(1, kernel_regularizer=reg_l2, bias_regularizer=reg_l2)(feat)
        y = tf.keras.layers.Activation('sigmoid')(y)

        epsilon = 0.001
        y = epsilon + y * (1-2*epsilon)

        model = tf.keras.Model(inputs=inputs, outputs = y, name='Discriminator')

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

def RRD_block(x, channel_nr=64, pad='SAME'):
    h1 = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=True, initialization=None)
    h1 = tf.keras.layers.LeakyReLU(alpha=0.2)(h1)
    h1 = tf.keras.layers.concatenate([h1, x])

    h2 = conv3d(h1, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=True, initialization=None)
    h2 = tf.keras.layers.LeakyReLU(alpha=0.2)(h2)
    h2 = tf.keras.layers.concatenate([h2, h1])

    h3 = conv3d(h2, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=True, initialization=None)
    h3 = tf.keras.layers.LeakyReLU(alpha=0.2)(h3)
    h3 = tf.keras.layers.concatenate([h3, h2])

    o = conv3d(h3, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)

    return x + o * 0.2