import keras
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split as sk_train_test_split
from HEOA import HEOA
from ESOA import ESOA
import E_HEOA_model as md
from keras.callbacks import CSVLogger



def create_dataset_from_raw(directory_path, resize_to, images_per_batch):
    resize_width = resize_to[0]
    resize_height = resize_to[1]
    batch_names = [os.path.join(directory_path, name) for name in os.listdir(directory_path) if
                   os.path.isdir(os.path.join(directory_path, name))]
    total_images = sum(len(os.listdir(batch_name)) for batch_name in batch_names)
    total_batches = -(-total_images // images_per_batch)
    print(total_batches)
    dataset = np.zeros(shape=(total_batches, images_per_batch, resize_height, resize_width, 1))
    batch_idx = 0
    for batch_path in batch_names:
        images = [os.path.join(batch_path, x) for x in os.listdir(batch_path) if x.endswith('.png')]
        images.sort()

        for i in range(0, len(images), images_per_batch):
            crn_batch = np.zeros(shape=(images_per_batch, resize_height, resize_width, 1))
            for idx, img_path in enumerate(images[i:i + images_per_batch]):
                img = Image.open(img_path).convert('L')
                img = img.resize(size=(resize_width, resize_height))
                img = np.array(img).astype(np.float32) / 255.0
                crn_batch[idx] = np.expand_dims(img, axis=-1)

            dataset[batch_idx] = crn_batch
            batch_idx += 1
            print(f"Importing batch: {batch_idx}")

    return dataset
shape = 64

dataset = create_dataset_from_raw(
    r'D:\学习文件\wpy\HKO-7Dataset\HKO-7\val', resize_to=(shape,shape), images_per_batch=20
)

def split_data_xy(data):
    x = data[:, :10, :, :, :]
    y = data[:, 10:, :, :, :]
    return x, y
print(type(dataset))
dataset_x, dataset_y = split_data_xy(dataset)

X_train, X_val, y_train, y_val = sk_train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=42)


model_param = {
    "test_data": X_val,
    "test_label":y_val,
    "data": X_train,
    "label": y_train
}
OA_param = {
    "size_pop": 4,
    "n_dim": 2,
    "max_iter": 2,
    "lb":np.array([0.1, 0.0001]),
    "ub":np.array([0.5, 0.01])
}
# # 人类进化优化算法（加变异）
# heoa = HEOA(model_param, OA_param)
# best_learn_rate,dropout, best_err = heoa.run()
# 白鹭群优化算法（加变异）
esoa = ESOA(model_param, OA_param)
best_learn_rate,dropout, best_err = esoa.run()



print(best_learn_rate)
print(dropout)
model = md.create_model(dropout=dropout)
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=best_learn_rate))

# csv_logger_path = 'LOSS_HEOA.csv'
csv_logger_path = 'LOSS_ESOA.csv'

csv_logger = CSVLogger(csv_logger_path, append=True, separator=',')

epochs = 30
batch_size = 1

# Fit the model
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[csv_logger]
)

# model.save('model_HEOA.h5')
model.save('model_ESOA.h5')