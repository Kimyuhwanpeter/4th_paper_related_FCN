# -*- coding:utf-8 -*-
import tensorflow as tf

def FCN(input_shape=(512, 512, 3), num_classes=3):

    pre_trained_VGG16 = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False)

    p1 = pre_trained_VGG16.get_layer("block1_pool").output
    p2 = pre_trained_VGG16.get_layer("block2_pool").output
    p3 = pre_trained_VGG16.get_layer("block3_pool").output
    p4 = pre_trained_VGG16.get_layer("block4_pool").output
    p5 = pre_trained_VGG16.get_layer("block5_pool").output

    c6 = tf.keras.layers.Conv2D(4096, (7,7), activation="relu", padding="same")(p5)
    c7 = tf.keras.layers.Conv2D(4096, (1,1), activation="relu", padding="same")(c6)

    f1, f2, f3, f4, f5 = p1, p2, p3, p4, c7

    # FCN-32 output
    fcn32_o = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(32,32), strides=(32, 32), use_bias=False)(f5)
    fcn32_o = tf.keras.layers.Activation('softmax')(fcn32_o)
    
    o = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(4,4), strides=(2,2), use_bias=False)(f5) # (16, 16, n)
    o = tf.keras.layers.Cropping2D(cropping=(1,1))(o) # (14, 14, n)
    
    o2 = f4 # (14, 14, 512)
    o2 = tf.keras.layers.Conv2D(num_classes, (1,1), activation='relu', padding='same')(o2) # (14, 14, n)
    
    o = tf.keras.layers.Add()([o, o2]) # (14, 14, n)
    # FCN-16 output
    fcn16_o = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(16,16), strides=(16,16), use_bias=False)(o)
    fcn16_o = tf.keras.layers.Activation('softmax')(fcn16_o)
    
    o = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(4,4), strides=(2,2), use_bias=False)(o) # (30, 30, n)
    o = tf.keras.layers.Cropping2D(cropping=(1,1))(o) # (28, 28, n)
    
    o2 = f3 # (28, 28, 256)
    o2 = tf.keras.layers.Conv2D(num_classes, (1,1), activation='relu', padding='same')(o2) # (28, 28, n)
    
    o = tf.keras.layers.Add()([o, o2]) # (28, 28, n)
    
    o = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(8,8), strides=(8,8), use_bias=False)(o) # (224, 224, n)
    
    fcn8_o = o
 
    return tf.keras.Model(inputs=pre_trained_VGG16.input, outputs=fcn16_o)

model = FCN()
model.summary()
