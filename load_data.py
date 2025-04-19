import uproot
import numpy as np
import awkward as ak


class NOvAData:
    def __init__(self, files,  merge_numu_quartiles):
        self.file_locations = files
        self.merge_numu_q: bool = merge_numu_quartiles

        self.zero_bins_nue = [0, 1, 8, 9, 10, 17, 18, 19, 21, 22]
        self.zero_bins_nueLowE = [0, 3]

        self.__get_data()

    def __get_data():
        all_true = []
        all_data = []

        for i, f in enumerate(self.file_locations):
            print(f"loading {i}/{len(files)}: {f}")
            file = uproot.open(f)

            # Get true parameters
            t_true = file["true_values"]
            true_values = t_true.arrays(["delta_cp", "sinsq_2th13", "sinsq_th23", "dm32"], library="np")
            theta = np.column_stack([
                true_values["delta_cp"],
                true_values["sinsq_2th13"],
                true_values["sinsq_th23"],
                true_values["dm32"]
            ])
            all_true.append(theta)

            # Get synthetic spectra
            t_data = file["synthetic_data"]

            arrays = t_data.arrays(t_data.keys(), library="ak")
            nue_fhc = ak.to_numpy(arrays["nue_fhc"])
            nue_rhc = ak.to_numpy(arrays["nue_rhc"])
            nueLowE_fhc = ak.to_numpy(arrays["nueLowE_fhc"])

            # Create mask for indices to keep
            all_bins_fhc = np.arange(nue_fhc.shape[1])
            all_bins_rhc = np.arange(nue_rhc.shape[1])
            all_bins_nueLowE = np.arange(nueLowE_fhc.shape[1])
            kept_fhc = np.setdiff1d(all_bins_fhc, zero_bins_nue)
            kept_rhc = np.setdiff1d(all_bins_rhc, zero_bins_nue)
            kept_nueLowEfhc = np.setdiff1d(all_bins_nueLowE, zero_bins_nueLowE)

            nue_fhc_clean = nue_fhc[:, kept_fhc]
            nue_rhc_clean = nue_rhc[:, kept_rhc]
            nueLowE_fhc_clean = nue_rhc[:, kept_nueLowEfhc]

            if merge_numu_quartiles:
                # Sum the numu_fhc and numu_rhc subcategories
                numu_fhc_sum = arrays["numu_fhc_1"] + arrays["numu_fhc_2"] + arrays["numu_fhc_3"] + arrays["numu_fhc_4"]
                numu_rhc_sum = arrays["numu_rhc_1"] + arrays["numu_rhc_2"] + arrays["numu_rhc_3"] + arrays["numu_rhc_4"]

                # Stack the resulting 4 arrays side by side
                merged = ak.concatenate([nue_fhc_clean, nueLowE_fhc_clean, nue_rhc_clean, numu_fhc_sum, numu_rhc_sum], axis=1)

            else:
                merged = ak.concatenate([arrays[key] for key in t_data.keys() ], axis=1)

            # Convert to numpy
            x_data = ak.to_numpy(merged)

            all_data.append(x_data)

        # Stack everything into one big array
        self.all_theta = np.vstack(all_true)
        self.all_spectra = np.vstack(all_data)

    def get_training_data(normalize=True,
                          train_fraction=0.8,
                          validation_fraction=0.1,
                          testing_fraction=0.1):
        # Normalize if requested
        if normalize:
            self.norm_spectra_mean = np.mean(self.all_spectra, axis=0)
            self.norm_spectra_std = np.std(self.all_spectra, axis=0) + 1e-6
            self.all_spectra = (self.all_spectra - self.norm_spectra_mean) / self.norm_spectra_std

            self.norm_theta_mean = np.mean(self.all_theta, axis=0)
            self.norm_theta_std = np.std(self.all_theta, axis=0) + 1e-6
            self.all_theta = (self.all_theta - self.norm_theta_mean) / self.norm_theta_std
        else:
            self.norm_spectra_mean = self.norm_spectra_std = self.norm_theta_mean = self.norm_theta_std = None

        # Optional shuffle (recommended!)
        total = len(all_x_data)
        indices = np.arange(total)
        np.random.shuffle(indices)

        # Determine sizes
        n_trn = int(total * train_fraction)
        n_val = int(total * validation_fraction)
        n_tes = int(total * testing_fraction)
        n_spectr = all_x_data.shape[1]


        all_x_data = all_x_data[indices]
        all_theta = all_theta[indices]

        # Return train, validation, test
        train = (all_x_data[:n_trn].reshape((-1, n_spectr, 1)), all_theta[:n_trn])
        val = (all_x_data[n_trn:n_trn + n_val].reshape((-1, n_spectr, 1)), all_theta[n_trn:n_trn + n_val])
        test = (all_x_data[n_trn + n_val:n_trn + n_val + n_tes].reshape((-1, n_spectr, 1)), all_theta[n_trn + n_val:n_trn + n_val + n_tes])

        return train, val, test, (self.norm_spectra_mean, self.norm_spectra_std), (self.norm_theta_mean, self.norm_theta_std)

def get_data(files,
             merge_numu_quartiles=True,
             validation_events=0.1,
             testing_events=0.1,
             spectra_norm=None,
             true_norm=None):

    #if (train_fraction + validation_fraction + testing_fraction > 1.0):
    #    raise ValueError(f"Train fraction ({train_fraction}), validation fraction ({validation_fraction}), and testing fraction ({testing_fraction}) add to more than 1!")

    zero_bins_nue = [0, 1, 8, 9, 10, 17, 18, 19, 21, 22]
    zero_bins_nueLowE = [0, 3]

    all_true = []
    all_data = []

    for i, f in enumerate(files):
        print(f"loading {i}/{len(files)}: {f}")
        file = uproot.open(f)

        # Get true parameters
        t_true = file["true_values"]
        true_values = t_true.arrays(["delta_cp", "sinsq_2th13", "sinsq_th23", "dm32"], library="np")

        true_values["mass_ordering"] = (true_values["dm32"] > 0).astype(int)
        true_values["dm32"] = np.abs(true_values["dm32"])

        theta = np.column_stack([
            true_values["delta_cp"],
            true_values["sinsq_2th13"],
            true_values["sinsq_th23"],
            true_values["dm32"],
            true_values["mass_ordering"]
        ])
        all_true.append(theta)

        # Get synthetic spectra
        t_data = file["synthetic_data"]

        arrays = t_data.arrays(t_data.keys(), library="ak")
        nue_fhc = ak.to_numpy(arrays["nue_fhc"])
        nue_rhc = ak.to_numpy(arrays["nue_rhc"])
        nueLowE_fhc = ak.to_numpy(arrays["nueLowE_fhc"])

        # Create mask for indices to keep
        all_bins_fhc = np.arange(nue_fhc.shape[1])
        all_bins_rhc = np.arange(nue_rhc.shape[1])
        all_bins_nueLowE = np.arange(nueLowE_fhc.shape[1])
        kept_fhc = np.setdiff1d(all_bins_fhc, zero_bins_nue)
        kept_rhc = np.setdiff1d(all_bins_rhc, zero_bins_nue)
        kept_nueLowEfhc = np.setdiff1d(all_bins_nueLowE, zero_bins_nueLowE)

        nue_fhc_clean = nue_fhc[:, kept_fhc]
        nue_rhc_clean = nue_rhc[:, kept_rhc]
        nueLowE_fhc_clean = nue_rhc[:, kept_nueLowEfhc]

        if merge_numu_quartiles:
            # Sum the numu_fhc and numu_rhc subcategories
            numu_fhc_sum = arrays["numu_fhc_1"] + arrays["numu_fhc_2"] + arrays["numu_fhc_3"] + arrays["numu_fhc_4"]
            numu_rhc_sum = arrays["numu_rhc_1"] + arrays["numu_rhc_2"] + arrays["numu_rhc_3"] + arrays["numu_rhc_4"]

            # Stack the resulting 4 arrays side by side
            merged = ak.concatenate([nue_fhc_clean, nueLowE_fhc_clean, nue_rhc_clean, numu_fhc_sum, numu_rhc_sum], axis=1)

        else:
            merged = ak.concatenate([arrays[key] for key in t_data.keys() ], axis=1)


        # Convert to numpy
        x_data = ak.to_numpy(merged)

        all_data.append(x_data)

    # Stack everything into one big array
    all_theta = np.vstack(all_true)
    all_x_data = np.vstack(all_data)

    print(all_x_data.shape)

        # Normalize if requested
    if spectra_norm is not None and true_norm is not None:
        all_x_data = (all_x_data - spectra_norm[0]) / spectra_norm[1]
        all_theta = (all_theta - true_norm[0]) / true_norm[1]
    else:
        spectra_mean = np.mean(np.mean(all_x_data, axis=1, keepdims=True))
        spectra_std = np.mean(np.std(all_x_data, axis=1, keepdims=True) + 1e-6)
        all_x_data = (all_x_data - spectra_mean) / spectra_std
        spectra_norm = (spectra_mean, spectra_std)

        theta_mean = np.mean(all_theta, axis=0)
        theta_std = np.std(all_theta, axis=0) + 1e-6
        all_theta = (all_theta - theta_mean) / theta_std
        true_norm = (theta_mean, theta_std)

    ## Optional shuffle (recommended!)
    total = len(all_x_data)
    #indices = np.arange(total)
    #np.random.shuffle(indices)

    if total - (validation_events + testing_events) <= 0:
        raise ValueError(f"Not enough events for training! Total: {total}, Validation: {validation_events}, Testing: {testing_events}")

    # Determine sizes
    n_trn = int(total - (validation_events + testing_events))
    n_val = int(validation_events)
    n_tes = int(testing_events)
    n_spectr = all_x_data.shape[1]


    #all_x_data = all_x_data[indices]
    #all_theta = all_theta[indices]

    # Return train, validation, test
    #train = (all_x_data[:n_trn].reshape((-1, n_spectr, 1)), all_theta[:n_trn])
    #val = (all_x_data[n_trn:n_trn + n_val].reshape((-1, n_spectr, 1)), all_theta[n_trn:n_trn + n_val])
    #test = (all_x_data[n_trn + n_val:n_trn + n_val + n_tes].reshape((-1, n_spectr, 1)), all_theta[n_trn + n_val:n_trn + n_val + n_tes])

    #return train, val, test, spectra_norm, true_norm

    return ((all_x_data[:n_trn].reshape((-1, n_spectr, 1)), all_theta[:n_trn]),
           (all_x_data[n_trn:n_trn + n_val].reshape((-1, n_spectr, 1)), all_theta[n_trn:n_trn + n_val]),
           (all_x_data[n_trn + n_val:n_trn + n_val + n_tes].reshape((-1, n_spectr, 1)), all_theta[n_trn + n_val:n_trn + n_val + n_tes]),
           spectra_norm, true_norm)

#def get_data(files,
#             train_fraction = 0.8,
#             validation_fraction = 0.1,
#             testing_fraction = 0.1):
#
#    if (train_fraction + validation_fraction + testing_fraction > 1.0):
#        raise ValueError(f"Train fraction ({train_fraction}), validation fraction ({validation_fraction}) and testing fraction ({testing_fraction}) add to more than 1!")
#
#    file = uproot.open(files[0])
#
#    t_true = file["true_values"]
#    true_values = t_true.arrays(["delta_cp", "sinsq_2th13", "sinsq_th23", "dm32"], library="np")
#    theta_train = np.column_stack([
#        true_values["delta_cp"],
#        true_values["sinsq_2th13"],
#        true_values["sinsq_th23"],
#        true_values["dm32"]
#    ])
#    t_true = theta_train
#
#    t_data = file["synthetic_data"]
#    keys = t_data.keys()
#    arrays = t_data.arrays(keys, library="ak")
#    merged = ak.concatenate([arrays[key] for key in keys], axis=1)
#    x_data = ak.to_numpy(merged)
#
#    n_trn = int(len(x_data)*train_fraction)
#    n_val = int(len(x_data)*validation_fraction)
#    n_tes = int(len(x_data)*testing_fraction)
#
#    n_spectr = len(x_data[0])
#
#    return (x_data[:n_trn].reshape((-1, n_spectr, 1)), t_true[:n_trn]), (x_data[n_trn:n_trn+n_val].reshape((-1, n_spectr, 1)), t_true[n_trn:n_trn+n_val]), (x_data[n_trn+n_val:n_trn+n_val+n_trn].reshape((-1, n_spectr, 1)), t_true[n_trn+n_val:n_trn+n_val+n_trn])

if __name__ == "__main__":
    files = [
            "/home/artur/work/nova/workspace/synthetic_data/synthetic_data_seed42_samples100000.root",
            "/home/artur/work/nova/workspace/synthetic_data/synthetic_data_seed43_samples100000.root",
            "/home/artur/work/nova/workspace/synthetic_data/synthetic_data_seed44_samples100000.root"
            ]
    train, validate, test, spectra_norm, val_norm = get_data(files)
