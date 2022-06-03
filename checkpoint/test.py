import numpy as np

b = np.load('Generator_Models/g_srgan.npz')
print(b['params'])

'''
InputLayer  SRGAN_g/in: (?, 122, 122, 3)
Conv2dLayer SRGAN_g/n64s1/c: shape:(3, 3, 3, 64) strides:(1, 1, 1, 1) pad:SAME act:relu

Conv2dLayer SRGAN_g/n64s1/c1/0: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/0: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/0: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/0: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/0: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/1: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/1: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/1: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/1: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/1: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/2: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/2: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/2: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/2: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/2: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/3: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/3: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/3: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/3: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/3: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/4: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/4: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/4: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/4: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/4: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/5: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/5: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/5: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/5: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/5: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/6: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/6: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/6: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/6: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/6: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/7: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/7: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/7: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/7: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/7: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/8: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/8: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/8: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/8: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/8: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/9: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/9: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/9: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/9: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/9: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/10: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/10: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/10: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/10: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/10: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/11: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/11: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/11: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/11: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/11: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/12: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/12: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/12: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/12: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/12: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/13: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/13: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/13: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/13: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/13: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/14: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/14: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/14: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/14: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/14: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c1/15: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b1/15: decay:0.900000 epsilon:0.000010 act:relu is_train:False
Conv2dLayer SRGAN_g/n64s1/c2/15: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b2/15: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/b_residual_add/15: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n64s1/c/m: shape:(3, 3, 64, 64) strides:(1, 1, 1, 1) pad:SAME act:identity
BatchNormLayer SRGAN_g/n64s1/b/m: decay:0.900000 epsilon:0.000010 act:identity is_train:False
ElementwiseLayer SRGAN_g/add3: size:(?, 122, 122, 64) fn:add

Conv2dLayer SRGAN_g/n256s1/1: shape:(3, 3, 64, 256) strides:(1, 1, 1, 1) pad:SAME act:identity
SubpixelConv2d  pixelshufflerx2/1: scale: 2 n_out_channel: 64 act: relu

Conv2dLayer SRGAN_g/n256s1/2: shape:(3, 3, 64, 256) strides:(1, 1, 1, 1) pad:SAME act:identity
SubpixelConv2d  pixelshufflerx2/2: scale: 2 n_out_channel: 64 act: relu

Conv2dLayer SRGAN_g/out: shape:(1, 1, 64, 3) strides:(1, 1, 1, 1) pad:SAME act:tanh
'''

