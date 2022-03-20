import os
import sys
import tensorflow as tf
from tensorflow.keras import Model, layers, models,\
    preprocessing, metrics, losses, optimizers, callbacks
import tensorflow_addons as tfa
import wandb
from wandb.keras import WandbCallback

import flwr as fl

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

config = {
    'num_clients': 2,
    'batch_size': 4,
    'num_rounds': 5,
    'epochs_per_round': 1,
    'validation_split': .2,
    'seed': 41
}


def start_client(client_id):

    def build_araujo_model() -> Model:
        preprocessing_layer = tf.keras.Sequential([
            layers.experimental.preprocessing.Rescaling(scale=1./255),
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")])

        model = models.Sequential()
        model.add(layers.Conv2D(16, 3, activation='relu',
                                kernel_initializer='he_uniform', input_shape=(512, 512, 3)))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(32, 3, activation='relu',
                                kernel_initializer='he_uniform'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, 3, activation='relu',
                                padding='same', kernel_initializer='he_uniform'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, 3, activation='relu',
                                padding='same', kernel_initializer='he_uniform'))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(32, 3, activation='relu',
                                kernel_initializer='he_uniform'))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu',
                               kernel_initializer='he_uniform'))
        model.add(layers.Dense(128, activation='relu',
                               kernel_initializer='he_uniform'))
        model.add(layers.Dense(4, activation='softmax'))

        inputs = tf.keras.Input(shape=(512, 512, 3))
        x = preprocessing_layer(inputs)
        outputs = model(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    def compute_mean_var(dataset: tf.data.Dataset):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for data, _ in dataset:
            channels_sum += tf.reduce_mean(data,
                                           axis=[0, 1, 2], keepdims=False)
            channels_squared_sum += tf.reduce_mean(data**2, axis=[0, 1, 2])
            num_batches += 1
        mean = channels_sum / num_batches
        var = channels_squared_sum/num_batches - mean**2
        #std = var**0.5
        return mean, var

    def preprocess(dataset, mean, variance):
        z_norm = layers.experimental.preprocessing.Normalization(
            mean=mean, variance=variance)

        def norm_fn(data, labels):
            return z_norm(data), labels
        return dataset.map(norm_fn)

    print("CLIENT #{}: Loading dataset...".format(client_id))
    dataset = (_load_directory(client_id, True),
               _load_directory(client_id, False))
    print("CLIENT #{}: Computing dataset mean and variance...".format(client_id))
    mean, variance = compute_mean_var(dataset[0])
    print("CLIENT #{}: Mean: {}, Variance: {}".format(client_id, mean, variance))
    train_dataset = preprocess(dataset[0], mean, variance)
    valid_dataset = preprocess(dataset[1], mean, variance)

    print("CLIENT #{}: Building the model...".format(client_id))
    model = build_araujo_model()
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss=losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[
                      metrics.CategoricalAccuracy(),
                      tfa.metrics.F1Score(num_classes=4, average='macro')])

    class BachClient(fl.client.NumPyClient):
        def get_parameters(self):
            return model.get_weights()

        def fit(self, parameters, config):
            print("CLIENT #{}: Starting local training...".format(client_id))
            model.set_weights(parameters)
            model.fit(train_dataset, epochs=1,
                      validation_data=valid_dataset,
                      verbose=2,  # single line per epoch
                      callbacks=[WandbCallback(monitor="val_loss")]
                      )
            return model.get_weights(), len(train_dataset), {}

        def evaluate(self, parameters, config):
            print("CLIENT #{}: Evaluating the model...".format(client_id))
            model.set_weights(parameters)
            loss, accuracy, f1_score = model.evaluate(valid_dataset, verbose=2)
            return loss, len(valid_dataset), {"accuracy": accuracy, "f1_score": f1_score}

    wandb.login()
    wandb.init(project="bach_fedavg", entity="xpetrov")
    wandb.config = config

    print("CLIENT #{}: Starting the client...".format(client_id))
    fl.client.start_numpy_client("127.0.0.1:8080", client=BachClient())


def _load_directory(id: int, train: bool):
    subset = "training" if train else "validation"
    shuffle = True if train else False
    return preprocessing.image_dataset_from_directory(
        'D:\\BACH_dataset\\ICIAR2018_BACH_Challenge\\Photos_balanced_sites_png\\site' +
        str(id),
        label_mode='categorical',
        class_names=["Benign", "InSitu", "Invasive", "Normal"],
        batch_size=config['batch_size'],
        image_size=(512, 512),
        shuffle=shuffle,
        validation_split=config['validation_split'],
        seed=config['seed'],
        subset=subset
    )


if __name__ == '__main__':
    start_client(client_id=sys.argv[1])
