import bayesflow as bf
import tensorflow as tf
#from tensorflow.keras import layers
import keras
import numpy as np

from load_data import get_data
import matplotlib.pyplot as plt


"""

# Data manipulations
1. Merge numu spectra (done!)
2. Super-normalize the data (nue, numu, rhc, fhc) (no... done differently!)
3. Remove nue zero bins (done!)
4. Change dm32 to abs(dm32) & mo (binary) (done!)

5. Try with more data
6. Try with expectrations rather than predictions

# Network manipulations
1. Maybe try making predictions. Perhaps looks ok?
2. Try Time Series network! (done!)
3. Try MLP (done!)
4. Try DeepSet "hacked" to be 1xX D (done!)
5. Split spectra into the 5 samples, and use stacked DeepSet
6. Split spectra into the 5 samples, and use tokenized Transformer (difficult...)
"""

class CustomMLPSummaryNetwork(bf.networks.SummaryNetwork):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)

        self.mlp = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation='gelu'),
            keras.layers.Dense(64, activation='gelu'),
            keras.layers.Dropout(0.05),
        ])
        self.summary_stats = keras.layers.Dense(16)

    def call(self, x, **kwargs):
        summary = self.mlp(x, training=kwargs.get("stage") == "training")
        summary = self.summary_stats(summary)
        return summary

def plot_boxes(g, boxes, dataset_id, color="blue"):
    for i,(key, box) in enumerate(boxes.items()):
        for j in range(4):
            g.axes[j,i].axvline(box[dataset_id,0,0], color=color, linestyle=":")
            g.axes[j,i].axvline(box[dataset_id,1,0], color=color, linestyle=":")
            if i != j:
                g.axes[i,j].axhline(box[dataset_id,0,0], color=color, linestyle=":")
                g.axes[i,j].axhline(box[dataset_id,1,0], color=color, linestyle=":")

def get_approximator(summary_network_type="time_series"):
    summary_net = None

    if summary_network_type == "deep_set":
        summary_net =bf.networks.DeepSet(
                summary_dim=256, 
                depth=4,
                set_embedding_dim=128,
                spectral_normalization=True
            )
    elif summary_network_type == "time_series":
        summary_net = bf.networks.TimeSeriesNetwork(
                conv_filters=[64, 128, 256],  # Try deeper if needed
                kernel_sizes=[1, 1, 1],
                dense_layers=[256, 128, 64],
                summary_dim=64
            )

    inference_net = bf.networks.FlowMatching()

    approximator = bf.approximators.ContinuousApproximator(
            inference_network=inference_net,
            summary_network=summary_net,
            adapter=bf.BasicWorkflow.default_adapter(
                inference_variables=["parameters"],
                summary_variables=["observables"],
                inference_conditions=None, 
                standardize=None,
            )
    )

    return approximator

def get_trained_model(model_file):

    approximator = keras.saving.load_model(model_file)

    return approximator 

def train_model(approximator, training_data, validation_data, epochs=10):

    approximator.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4)
                         #loss=keras.losses.Poisson()
                         )
    batch_size = 2048

    offline_dataset = bf.datasets.OfflineDataset(training_data, batch_size=batch_size, adapter=approximator.adapter)
    offline_dataset_validation = bf.datasets.OfflineDataset(validation_data, batch_size=batch_size, adapter=approximator.adapter)

    history = approximator.fit(dataset=offline_dataset, validation_data=offline_dataset_validation, epochs=epochs)
    
    return history