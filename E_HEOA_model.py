from keras import Sequential, Input, Model
from keras.layers import ConvLSTM2D, Dropout, BatchNormalization, Conv3D, MaxPooling3D, Flatten, Dense, Conv2D, \
    Conv3DTranspose, Activation, Add


def create_model(dropout):
    inputs = Input(shape=(10, 64, 64, 1))

    # 添加输入层和第一个Conv3D层
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(inputs)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(x)

    # 添加ConvLSTM2D层
    x = ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(x)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(x)

    outputs = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

