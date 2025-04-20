# Amortized Bayesian Inference using NOvA

A small proof of concept + tests for Amortized Bayesian Inference using a bit more complex datasets than what's normally provided in the literature.
Using NOvA MC to train an Amortized model to see if we can train on FHC + RHC, numu + nue data with all osc + systematic parameters (~50 in total).

## Tasks

- [ ] Show ABI feasible for complex models like NOvA
  - [ ] Stats-only
  - [ ] With systematics
- [ ] Create out-of sample tests to further validate the amortized model.
- [ ] Look into ABI accuracy by evaluating Simulation-Based Calibration diagnostics to check for convergence: goodbye p-values!
- [ ] Implement Sensitivity-Aware ABI (different priors, try fake datasets if available)

## Tests

---

### 1. `notebooks/train_model_emulated.ipynb`

Training amortized NOvA model on emulated data:

1. Emulated data: 10M **expectations** for nueLowE FHC, nue FHC, nue RHC, numu FHC & numu RHC (created for all numu quartiles & merged into one FHC & one RHC spectra), created for `delta_cp`, `sinsq 2theta13`, `sinsq theta23`, `dm32` and `mass_ordering` drawn from uniform priors (Gaussian for th13).

2. Successfully trained an amortized model with those expectations.
3. Diagnostics show great computing performance (as expected), good parameter recovery on testing datasets, but strong overfitting: expected due to the use of **expectations** rather than **predictions**.

**Steps for the next notebook**:

1. Re-do using **predictions** rather than **expectations** (so poisson-fluctuated expectations).
2. Include all the systematic parameters. This could be done in two ways.
    1. Draw systematics from their priors when creating **predictions**, so the predictions are systematically fluctuated: but do not include systematic parameters in the "labels" or "true parameters". This will effectively marginalize our spectra over the systematic parameters and let network figure the systematic shifts by itself. Pros: far easier to deal with. Cons: cannot infer about systematic parameters.
    2. Draw the systematics from priors when creating predictions like above, but do include them in the "labels" or "true parameters" during training. This will increase the number of labels from 5 to ~60? Pros: can infer about the systematic parameters. Cons: more difficult, might require more complex network.
3. Include the POT as one of the parameters to vary when creating simulations! This means the amortized model could be used for future sensitivity studies. POT could be added into the network in two ways:
    1. Add POT as "inference_conditions" parameters to the network. This is probably the more correct way.
    2. Add POT as two extra bins to our merged spectrum / input data vector. This *might* work too, and is probably easier than above.
4. See if a simpler network could suffice: reduce the layer sizes & number of layers, increase dropout.

---
