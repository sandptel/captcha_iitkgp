import os
import git
if os.path.exists("../captcha_iitkgp/") is not True:
    repo = git.Repo.clone_from("https://github.com/Sandeep-1507/Captcha_iitkgp", "./")

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math, re, os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE

# Detect TPU, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

data_dir=Path("./Data/trainset")
images=sorted(list(map(str,list(data_dir.glob("*.jpeg")))))



display(images[0])

labels=[img.split(os.path.sep)[-1].split(".jpeg")[0] for img in images]

# print(labels)

characters=set(char for label in labels for char in label)
print(characters)
characters=sorted(list(characters))


# print(len(characters))
# images=[clear_img(img) for img in images]

batch_size = 24
sample=Image.open(images[0])
image_width,image_height=(198,50)
print(image_width,image_height)

downsample_factor=2
max_length = max([len(label) for label in labels])
display(max_length)

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup
char_to_num=layers.StringLookup(vocabulary=list(characters),mask_token=None)

num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def encode_single_sample(img_path,label):
    img=tf.io.read_file(img_path)
    img=tf.io.decode_jpeg(img,channels=1)
    img=tf.image.convert_image_dtype(img,tf.float32)
#     img=tf.transpose(img)
#     img=tf.image.resize(img,[image_width,image_height])
#     img=tf.image.resize(img,[sample])
#     img = tf.transpose(img, perm=[1, 0, 2]) #????
    label=char_to_num(tf.strings.unicode_split(label,input_encoding="UTF-8"))

    return {"image": img, "label": label}

print(encode_single_sample(images[0],labels[0]))

def split_data(images,labels,train_size=0.9,shuffle=True):
    size=len(images)
    indices=np.arange(size)
    if(shuffle):
        np.random.shuffle(indices)
    train_samples= int(size*train_size)

    x_train,y_train=images[indices[:train_samples]],labels[indices[:train_samples]]
    x_val,y_val=images[indices[train_samples:]],labels[indices[train_samples:]]

    return x_train,y_train,x_val,y_val

x_train, y_train, x_val, y_val = split_data(np.array(images), np.array(labels))

x_train.size

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
validation_dataset = (validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

# display(train_dataset[0])

_,ax=plt.subplots(4,4,figsize=(10,5))
for batch in train_dataset.take(1):
    images = batch["image"]
    labels = batch["label"]
    for i in range(16):
        img=(images[i]*255).numpy().astype("uint8")
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
        ax[i//4, i % 4].imshow(img[:, :, 0], cmap="gray",aspect="1")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")

plt.show()

#single sample
image=batch["image"][0]
label=batch["label"][0]
img=(image*255).numpy().astype("uint8")
labl= tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
image = Image.fromarray(img.squeeze())
plt.imshow(image,cmap="gray",aspect="1")
print(labl)
print(img.shape)

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):

        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)


        return y_pred
    
def build_model():

    input_img = layers.Input(
        shape=(image_height,image_width, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")


    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)


    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)


    new_shape = ((image_height // 4), (image_width // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)


    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)


    output = CTCLayer(name="ctc_loss")(labels, x)


    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )

    opt = keras.optimizers.Adam()

    model.compile(optimizer=opt)
    return model



model = build_model()
model.summary()

epochs = 100
early_stopping_patience = 10

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]

    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

for batch in validation_dataset.take(4):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)


    from sklearn.metrics import accuracy_score

    accuracy=accuracy_score(pred_texts,orig_texts)*100
    print("accuracy of prediction = {}".format(accuracy))



    _, ax = plt.subplots(6, 4, figsize=(15, 7))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        # img = img
        title = f"Prediction: {pred_texts[i]}"
        if i < 24:
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
plt.show()

# model.save("/content/Captcha_iitkgp/model")

# !zip -r file.zip /content/Captcha_iitkgp/model

# from IPython.display import FileLink
# FileLink(r'file.zip')

# len(x_val)