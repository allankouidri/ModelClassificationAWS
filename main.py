#load the data
#unpack the data
#create the data generator
#create training set / validation set
#train the model

import data.raw.load_data as loader
import models.cnn as classifier
import os
import tensorflow as tf
from tensorflow import keras

src = 'https://dsti-aws-class-ak.s3.eu-west-1.amazonaws.com/kagglecatsanddogs_3367a.zip'
dst = './data/raw'

loader.get_data(src,dst)
loader.unzip_data(dst)

#clean the data
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(dst,"PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)


image_size = (180, 180)
batch_size = 2

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./data/raw/PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337, 
    #important to specify the seed to make sure training and validation data are exclusive (no corruption)
    #because they are coming from the same directory
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./data/raw/PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

model = classifier.make_model(input_shape=image_size + (3,), num_classes=2)

epochs = 1

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

TRACKING_URI = "http://52.215.94.6:5000/"
mlflow.tracking.set_tracking_uri(TRACKING_URI)
mlflow.start_run()

