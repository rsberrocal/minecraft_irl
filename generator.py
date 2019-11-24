import tensorflow as tf
import variables
from tensorflow import keras
def encoder_layer(filters, apply_batchnorm = True ):

    encoded = keras.Sequential()

    initializer = tf.random_normal_initializer(0, 0.02)

    # Capa convolucional
    encoded.add(keras.layers.Conv2D(
                            filters,           #Cantidad filtros
                            kernel_initializer = initializer, #Inizializador de ruido de fitlro
                            kernel_size = 4,   #Tamaño de los filtros
                            strides = 2,       #Cantidad de "pasos" que se mueve el filtro
                            padding = "same",   #??
                            use_bias = not apply_batchnorm)
                )

    if apply_batchnorm:

        # Capa BatchNormalization
        encoded.add(keras.layers.BatchNormalization())

    # Capa Activacion(LeakyReLU)
    encoded.add(keras.layers.LeakyReLU())

    return encoded

def decoder_layer(filters, apply_dropout = True):

    decoded = keras.Sequential()

    initializer = tf.random_normal_initializer(0, 0.02)

    # Capa convolucional inversa
    decoded.add(keras.layers.Conv2DTranspose(
                                            filters,           #Cantidad filtros
                                            kernel_initializer = initializer, #Inizializador de ruido de fitlro
                                            kernel_size = 4,   #Tamaño de los filtros
                                            strides = 2,       #Cantidad de "pasos" que se mueve el filtro
                                            padding = "same",   #??
                                            use_bias = False
                                            )
                )
    #Capa de BatchNormalitzation
    decoded.add(keras.layers.BatchNormalization())


    #Capa de DropOut(previene el overfitting porque quita "dependencia"
    if apply_dropout:
        decoded.add(keras.layers.Dropout(0.5))


    # Capa Activacion(LeakyReLU)
    decoded.add(keras.layers.LeakyReLU())

    return decoded

def generator():
    #Definimos como será la layer de entrada (Ancho, Alto, Codificacion colores(RGB == 3))
    inputs = tf.keras.layers.Input(shape= [variables.WIDTH,variables.HEIGHT,3])

    encode_stack = [

                                                        # (cantidad_imagenes, 512, 512, nª filtros
                                                        # (en este caso 3 por el rgb)
                                                        # input a la primera capa
        #En la primera capa del encoder                 O utputs:
            encoder_layer(64, apply_batchnorm=False),   # (bs, 256, 256, 64)

            encoder_layer(128),                         # (bs, 128, 128, 128)
            encoder_layer(256),                         # (bs, 64, 64, 256)
            encoder_layer(512),                         # (bs, 32, 32, 512)
            encoder_layer(1024),                        # (bs, 16, 16, 1024)
            encoder_layer(1024),                        # (bs, 8, 8, 1024)
            encoder_layer(1024),                        # (bs, 4, 4, 1024)
            encoder_layer(1024),                        # (bs, 2, 2, 1024)
            encoder_layer(1024),                        # (bs, 1, 1, 1024)
    ]
    decode_stack = [

            decoder_layer(1024),                        # (bs, 2, 2, 1024)
            decoder_layer(1024),                        # (bs, 4, 4, 1024)
            decoder_layer(1024),                        # (bs, 8, 8, 1024)
            decoder_layer(1024),                        # (bs, 16, 16, 1024)
            decoder_layer(512),                         # (bs, 32, 32, 512)
            decoder_layer(256),                         # (bs, 64, 64, 256)
            decoder_layer(128),                         # (bs, 128, 128, 128)
            decoder_layer(64),                          # (bs, 256, 256, 64)
    ]

    initializer = tf.random_normal_initializer(0, 0.02)

    generation = keras.layers.Conv2DTranspose(
                                                filters = 3,
                                                kernel_size = 4,
                                                strides = 2,
                                                padding = "same",
                                                kernel_initializer= initializer,
                                                activation= "tanh"
                                             )
    x = inputs
    S = []
    concat = keras.layers.Concatenate()
    for encode in  encode_stack:

        x = encode(x)
        S.append(x)

    S = reversed(S[:-1])

    for decode, sk in zip(decode_stack, S):

        x = decode(x)
        concat = ([x, sk])
    generation=  generation(x)

    return keras.Model(inputs = inputs, outputs = generation)
