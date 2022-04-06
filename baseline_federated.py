import os
from random import shuffle
import time
from multiprocessing import Process
from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers, models,\
    preprocessing, metrics, losses, optimizers, utils
import tensorflow_addons as tfa

import flwr as fl
from flwr.common import Weights, Scalar, weights_to_parameters
from flwr.server.strategy import FedAvg, QFedAvg

import wandb
from wandb.keras import WandbCallback

from qfedadam import QFedAdam

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
WANDB_KEY = '70fc9e3697c791bb8e0bd252af2721808872bce9'

config = {
    'q_param': 0.2,
    'num_clients': 3,
    'batch_size': 4,
    'num_rounds': 30,
    'epochs_per_round': 1,
    #'validation_split': .2,
    'seed': 41,
    'dropout_fc_1': .25,
    'dropout_fc_2': .25,
    'f1_average': 'macro',
    'centralizaed_eval': False,
    'wandb_project': 'balanced_qfedavg',
    'wandb_entity': 'xpetrov'
}

DATA_ROOT = 'D:\\xpetrov\\ICIAR2018_BACH_Challenge\\'


def load_concrete_dataset(site_id: int, subset: str) -> tf.data.Dataset:
    shuffle = True if subset=='train' else False
    return preprocessing.image_dataset_from_directory(
        DATA_ROOT + 'federated\\balanced\\site' + str(site_id) + '\\' + subset,
        label_mode='categorical',
        class_names=["Benign", "InSitu", "Invasive", "Normal"],
        batch_size=config['batch_size'],
        image_size=(512, 512),
        shuffle=shuffle,
        seed=config['seed']
    )


def load_dataset(site_id: int) -> Tuple[
    tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    return (
        load_concrete_dataset(site_id, 'train'),
        load_concrete_dataset(site_id, 'valid'),
        load_concrete_dataset(site_id, 'test')
    )


def compute_mean_var(dataset: tf.data.Dataset) -> Tuple[float, float]:
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


def get_data_distribution(dataset):
    hist = tf.constant([0, 0, 0, 0], dtype=tf.int32)
    for _, labels in dataset:
        labels_flattened = tf.cast(tf.argmax(labels, axis=1), dtype=tf.int32)
        hist = tf.add(
            hist,
            tf.histogram_fixed_width(labels_flattened, [0, 3], nbins=4))
    hist_normalized = tf.divide(hist, tf.reduce_sum(hist))
    return hist.numpy(), hist_normalized.numpy()


def preprocess(dataset: tf.data.Dataset, mean: float, variance: float):
    z_norm = layers.experimental.preprocessing.Normalization(
        mean=mean, variance=variance)

    def norm_fn(data, labels):
        return z_norm(data), labels
    return dataset.map(norm_fn)


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
    model.add(layers.Dropout(rate=config["dropout_fc_1"]))
    model.add(layers.Dense(256, activation='relu',
                           kernel_initializer='he_uniform'))
    model.add(layers.Dropout(rate=config["dropout_fc_2"]))
    model.add(layers.Dense(128, activation='relu',
                           kernel_initializer='he_uniform'))
    model.add(layers.Dense(4, activation='softmax'))

    inputs = tf.keras.Input(shape=(512, 512, 3))
    x = preprocessing_layer(inputs)
    outputs = model(x)
    model = Model(inputs, outputs)
    return model


def start_server(num_rounds, num_clients) -> None:
    model = None
    if config['centralizaed_eval']:
        model = build_araujo_model()
        model.compile(optimizer=optimizers.Adam(1e-4),
                    loss=losses.CategoricalCrossentropy(from_logits=False),
                    metrics=[
                        metrics.CategoricalAccuracy(),
                        tfa.metrics.F1Score(num_classes=4, average=config['f1_average'])])
    
    def fit_config(rnd: int):
        return config.copy()

    def get_eval_fn(model: Model):
        """Return an evaluation function for server-side evaluation."""
        if config['centralizaed_eval'] is False:
            print("SERVER: No evaluation defined. Skip dataset loading.")
            return None
        dataset_id = config['num_clients']+1
        if dataset_id > 4:
            raise ValueError(f"No available dataset with id={dataset_id}")
        print("SERVER: Loading dataset...")
        dataset = (load_concrete_dataset(num_clients+1, 'train'),
                   load_concrete_dataset(num_clients+1, 'valid'))

        dist, dist_normed = get_data_distribution(dataset[1])
        print(f"SERVER: Validation Data distribution - {dist} ({dist_normed})")

        print("SERVER: Computing mean and variance...")
        mean, variance = compute_mean_var(dataset[0])
        print("SERVER: Mean: {}, Variance: {}".format(mean, variance))
        valid_dataset = preprocess(dataset[1], mean, variance)

        def evaluate(weights: Weights) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            model.set_weights(weights)
            metrics_dict = model.evaluate(
                valid_dataset, verbose=2, return_dict=True)
            for _ in range(config['epochs_per_round']-1):
                # log an empty dict *epr-1* times to increase wandb step
                # in order to allign server's log graph with clients' logs
                wandb.log({})
            wandb.log({
                'val_loss': metrics_dict['loss'],
                'val_categorical_accuracy': metrics_dict['categorical_accuracy'],
                'val_f1_score': metrics_dict['f1_score']
            })
            return metrics_dict['loss'], metrics_dict

        return evaluate

    strategy = QFedAvg(
        q_param=config["q_param"],
        min_available_clients=num_clients,
        #fraction_fit=1.0,
        #fraction_eval=1.0,
        min_fit_clients=num_clients,
        min_eval_clients=num_clients,
        on_fit_config_fn=fit_config,
        eval_fn=None if model is None else get_eval_fn(model),
        #initial_parameters=weights_to_parameters(
        #    model.get_weights()),
        )

    if (config['centralizaed_eval'] is True):
        wandb.login(key=WANDB_KEY)
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'])
        wandb.config = config

    print("SERVER: Starting the server...")
    # Exposes the server by default on port 8080
    fl.server.start_server(
        strategy=strategy,
        config={"num_rounds": num_rounds})


def start_client(client_id) -> None:

    print("CLIENT #{}: Loading dataset...".format(client_id))

    dataset = load_dataset(client_id)
    print("CLIENT #{}: Computing mean and variance...".format(client_id))
    mean, variance = compute_mean_var(dataset[0])
    print("CLIENT #{}: Mean: {}, Variance: {}".format(client_id, mean, variance))
    train_dataset = preprocess(dataset[0], mean, variance)
    valid_dataset = preprocess(dataset[1], mean, variance)
    #test_dataset = preprocess(dataset[2], mean, variance)

    print("CLIENT #{}: Building the model...".format(client_id))
    model = build_araujo_model()
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss=losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[
                      metrics.CategoricalAccuracy(),
                      tfa.metrics.F1Score(num_classes=4, average=config['f1_average'])])

    class BachClient(fl.client.NumPyClient):

        def get_parameters(self):
            return model.get_weights()

        def fit(self, parameters, config):
            print("CLIENT #{}: Starting local training...".format(client_id))
            model.set_weights(parameters)
            history = model.fit(train_dataset, epochs=config["epochs_per_round"],
                                validation_data=valid_dataset,
                                verbose=2,  # single line per epoch
                                callbacks=[WandbCallback()]
                                )
            results = {
                "loss": history.history["loss"][0],
                "categorical_accuracy": history.history["categorical_accuracy"][0],
                "f1_score": history.history["f1_score"][0],
                "val_loss": history.history["val_loss"][0],
                "val_categorical_accuracy": history.history["val_categorical_accuracy"][0],
                "val_f1_score": history.history["val_f1_score"][0]
            }
            return model.get_weights(), len(train_dataset), results

        def evaluate(self, parameters, config):
            print(
                "CLIENT #{}: Evaluating the model after aggregation...".format(client_id))
            model.set_weights(parameters)
            metrics_dict = model.evaluate(valid_dataset, verbose=2,
                                          return_dict=True,
                                          callbacks=[WandbCallback()]
                                          )
            print(metrics_dict)
            return metrics_dict['loss'], len(valid_dataset), metrics_dict

    wandb.login(key=WANDB_KEY)
    wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'])
    wandb.config = config

    print("CLIENT #{}: Starting the client...".format(client_id))
    fl.client.start_numpy_client("127.0.0.1:8080", client=BachClient())


def run_simulation(num_rounds, num_clients):
    processes = []

    server_process = Process(
        target=start_server, args=(num_rounds, num_clients)
    )
    server_process.start()
    processes.append(server_process)

    sleeptime = 60 if config['centralizaed_eval'] is True else 30
    time.sleep(sleeptime)  # should be enough for the server to initialize

    for client_id in range(1, num_clients+1):
        client_process = Process(
            target=start_client, args=(client_id,)
        )
        client_process.start()
        processes.append(client_process)

    for p in processes:
        p.join()


if __name__ == "__main__":
    run_simulation(
        num_rounds=config['num_rounds'],
        num_clients=config['num_clients'])
