# TODO: Create the 1st cell of USING_Model.ipynb

# import os
# if os.path.exists("./Model") is True:
#   !unzip ./Model/model.zip

import os
import zipfile
import git
if os.path.exists("../captcha_iitkgp/") is not True:
    repo = git.Repo.clone_from("https://github.com/Sandeep-1507/Captcha_iitkgp", "./")
if os.path.exists("./kaggle") is not True:
    if not os.path.exists("./"):
        os.makedirs("./")
    with zipfile.ZipFile("./Model/model.zip", 'r') as zip_ref:
        zip_ref.extractall("./")
    


import os
import numpy as np
import matplotlib.pyplot as plt
import string
from PIL import Image
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, save_model, load_model

model = load_model("./kaggle/working", compile = True)

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()


characters=set({ 'H', 'E', 'B', '8', 's', '2', 'J', 'K', 'G', 'T', '5', 'F', 'Y', 'U', 'C', 'R', 'L', 'S', 'W', 'Z', 'X', '9', '3', '7', '6', 'A', 'V', 'N', 'w', 'P', 'D', 'u', 'M', '4', 'c', 'x', 'y', 'j''C', 'T', '7', 'R', '5', '4', '2', 'D', 'M', 'G', 'Z', 'X', 'H', 'V', 'W', 'P', 'A', 'U', 'J', 'K', '6', 'B', 'N', 'E', '8', 'Y', '3', '9', 'S', 'F', 'L'})
print(characters)
characters=sorted(list(characters))
char_to_num=layers.StringLookup(vocabulary=list(characters),mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


def decode_batch_predictions(pred):
    max_length=6
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]

    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def captchasolve(folder_path,model,num_images=1):
    data_dir=Path(folder_path)
    images=sorted(list(map(str,list(data_dir.glob("*.jpeg")))))
    labels=[img.split(os.path.sep)[-1].split(".jpeg")[0] for img in images]
    # display(images[0])
    # display(labels[0])


    def encode_single_sample(img_path,label):
        img=tf.io.read_file(img_path)
        img=tf.io.decode_jpeg(img,channels=1)
        img=tf.image.convert_image_dtype(img,tf.float32)
        label=char_to_num(tf.strings.unicode_split(label,input_encoding="UTF-8"))

        return {"image": img, "label": label}

    max_length=6
    batch_size=num_images
    x_test, y_test = (np.array(images), np.array(labels))

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = (test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

    # batch_images = test_dataset.take(1)["image"]
    batch_images=[batch["image"] for batch in test_dataset.take(1)]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)
    print(pred_texts)

print("For testdataset: ")
captchasolve("./Data/testset",prediction_model,num_images=100)


print("For user provided captcha: ")
captchasolve("./Data/userset",prediction_model,num_images=100)
