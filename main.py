import bayesflow as bf
from bayesflow.diagnostics import plots as bf_plots

import tensorflow as tf
#from tensorflow.keras import layers
import keras
import numpy as np

from load_data import get_data
from train_model import get_trained_model, train_model, get_approximator
import matplotlib.pyplot as plt

import glob
import os


"""markdown

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

# Other
1. Do the transformer above
2. Generate high stats synthetic data
3. Generate data with varrying proton on target, and set that as inference condition
4. Try different loss functions (Poisson?)
5. Write a bunch of plotting scripts & posterior-saving
6. Write validation scripts showing comparisons against asimov A contours
7. Write an analysis for future POT sensitivity with NOvA
"""

if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description="BayesFlow example script")
    parser.add_argument("--train", action="store_true", help="Train the model.", default=False)
    parser.add_argument("--load", action="store_true", help="Load the model.", default=False)
    parser.add_argument("--model-name", type=str, help="Model name to load.", default="test_model_lstm")
    parser.add_argument("--test-sample", type=int, help="Number of test samples to use.", default=2)
    parser.add_argument("--diagnostics", action="store_true", help="Run diagnostics.", default=False)
    parser.add_argument("--data-files", type=str, nargs="+", help="Input data files (wildcards allowed).")

    args = parser.parse_args()

    spectra_norm = None
    true_norm = None
    if args.load:
        normalization = np.load(f"checkpoints/{args.model_name}_normalization.npz")
        spectra_norm = normalization["spectra_norm"]
        true_norm = normalization["true_norm"]
    
    data_files = []
    for data_file in args.data_files:
        data_files.extend(glob.glob(data_file))

    train, validate, test, spectra_norm, true_norm = get_data(data_files, 
                                                              validation_events=100_000,
                                                              testing_events=10_000,
                                                              spectra_norm=spectra_norm,
                                                              true_norm=true_norm)
                                                              
    print(f"Spectra normalisation: {spectra_norm}, input par norm: {true_norm}")

    n_params = len(train[1][0])
    n_spectr = len(train[0][0])

    training_data= {
            "parameters": train[1].astype(np.float32),
            "observables": train[0].astype(np.float32)
            }

    validation_data = {
            "parameters": validate[1].astype(np.float32),
            "observables": validate[0].astype(np.float32)
            }

    testing_data = {
            "parameters": test[1].astype(np.float32),
            "observables": test[0].astype(np.float32)
            }

    testing_data_sample = {
            "parameters": test[1][:args.test_sample].astype(np.float32),
            "observables": test[0][:args.test_sample].astype(np.float32)
            }

    approximator = None

    output_dir = f"diagnostics/{args.model_name}/"
    os.makedirs(output_dir, exist_ok=True)

    if args.train:
        print("Training model...")

        # Get the approximator
        approximator = get_approximator()

        # Train (fit) the model
        history = train_model(approximator, training_data=training_data, validation_data=validation_data, model_name=args.model_name)

        fig = bf.diagnostics.plots.loss(history)
        fig.savefig(f"diagnostics/{args.model_name}/loss.pdf")

        # Save the model
        approximator_checkpoint = f"checkpoints/{args.model_name}_approximator.keras"
        approximator.save(approximator_checkpoint)

        # Save the normalization parameters
        np.savez(f"checkpoints/{args.model_name}_normalization.npz", spectra_norm=spectra_norm, true_norm=true_norm)
        print("Model trained!")
    
    if args.load:
        approximator = get_trained_model(f"checkpoints/{args.model_name}_approximator.keras")
        print("loaded spectra normalization: ", spectra_norm)
        print("loaded true parameter normalization: ", true_norm)
        print("spectra normalization shape: ", spectra_norm.shape)

    variable_names = ["delta_cp", "th13", "th23", "dm32", "mass_ordering"]
    if args.diagnostics:
        test_posterior = approximator.sample(conditions=testing_data, num_samples=100)

        plot_fns = {
            "recovery": bf_plots.recovery,
            "calibration_ecdf": bf_plots.calibration_ecdf,
            "z_score_contraction": bf_plots.z_score_contraction,
            "calibration_histogram": bf_plots.calibration_histogram
        }

        figures = dict()

        for k, plot_fn in plot_fns.items():
            if  k == "calibration_ecdf":
                figures[k] = plot_fn(
                    estimates=test_posterior,
                    targets=testing_data,
                    variable_names=variable_names,
                    difference=True,
                    rank_type="distance"
                )
            else:
                figures[k] = plot_fn(
                    estimates=test_posterior,
                    targets=testing_data,
                    variable_names=variable_names
                )
            fig_path = os.path.join(output_dir, f"{k}.pdf")
            figures[k].savefig(fig_path)
            print(f"Saved {k} plot to {fig_path}")
            plt.close(figures[k])
    
    if args.test_sample > 0:
        print("sampling!")
        test_posterior = approximator.sample(conditions=testing_data_sample, num_samples=10_000)
        std = true_norm[1]
        mean = true_norm[0]
        test_posterior['parameters'] = test_posterior['parameters'] * std[None, None, :] + mean[None, None, :]
        testing_data_sample['parameters'] = testing_data_sample['parameters'] * std[None, :] + mean[None, :]
        print("sampling done!")

        dataset_id=0
        for n in range(args.test_sample):
            print(f"Plotting pairs posterior {n}")
            g = bf_plots.pairs_posterior(
                estimates=test_posterior, 
                targets=testing_data_sample,
                dataset_id=n,
                variable_names=["delta_cp", "th13", "th23", "dm32", "mass_ordering"],
            )
            plt.show()