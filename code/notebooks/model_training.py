import gc
import os
import random
import sys
import pickle
import json
import warnings
import numpy as np
import tensorflow as tf
import xarray as xr
from keras import Model
from keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D

# Setup Directories
MODEL_DIRECTORY = "../../data/saved_models/"
PREDICTIONS_DIRECTORY = "../../data/saved_predictions/"
DATA_DIRECTORY = "../../data/splits/"
DIAGNOSTICS_DIRECTORY = "../../data/model_diagnostics/"
FIGURES_DIRECTORY = "../../data/figures/"
NETCDF_DIRECTORY = "../../data/saved_netCDF/"

for directory in [MODEL_DIRECTORY, PREDICTIONS_DIRECTORY, DATA_DIRECTORY, 
                  DIAGNOSTICS_DIRECTORY, FIGURES_DIRECTORY, NETCDF_DIRECTORY]:
    os.makedirs(directory, exist_ok=True)

def get_settings(experiment_name):
    experiments = {
        "CanESM5_rsut": {
            "input_region": "X_train.nc", "detrend": True, "datafolder": "../../data/splits/",
            "anomalies": True, "anomalies_years": (0, 30), "data_period":"_1850-2100",
            "input_var": "tas", "label_var": "cre", "n_train_val_test": (17,4,4),
            "yr_bounds": (1850, 2014), "network_type": "cnn", "kernel_size": 3,
            "kernels": [32, 32], "ridge_param": [0.0, 0.0], "kernel_act": ["relu", "relu"],
            "hiddens": [32, 16], "act_fun": ["elu", "elu"], "learning_rate": 0.000005,
            "batch_size": 32, "rng_seed_list": np.arange(5).tolist(), "n_epochs": 25_000,
        },
        "CanESM5_rlut": {
            "input_region": "X_train.nc", "detrend": True, "datafolder": "../../data/splits/",
            "anomalies": True, "anomalies_years": (0, 30), "data_period":"_1850-2100",
            "input_var": "tas", "label_var": "cre", "n_train_val_test": (17,4,4),
            "yr_bounds": (1850, 2014), "network_type": "cnn", "kernel_size": 3,
            "kernels": [32, 32], "ridge_param": [0.0, 0.0], "kernel_act": ["relu", "relu"],
            "hiddens": [32, 16], "act_fun": ["elu", "elu"], "learning_rate": 0.000005,
            "batch_size": 32, "rng_seed_list": np.arange(5).tolist(), "n_epochs": 25_000,
        },
    }
    exp_dict = experiments[experiment_name]
    exp_dict['exp_name'] = experiment_name
    exp_dict['n_members'] = int(np.sum(exp_dict["n_train_val_test"]))
    return exp_dict

def compile_model(x_train, y_train, settings):
    inputs = Input(shape=x_train.shape[1:])
    layers = inputs
    if settings["network_type"] == "cnn":
        for kernel, kernel_act in zip(settings["kernels"], settings["kernel_act"]):
            layers = Conv2D(kernel, (settings["kernel_size"], settings["kernel_size"]),
                            activation=kernel_act, padding="same")(layers)
            layers = MaxPooling2D((2, 2))(layers)
        layers = Flatten()(layers)
        for hidden, activation in zip(settings["hiddens"], settings["act_fun"]):
            layers = Dense(hidden, activation=activation)(layers)
    
    output_layer = Dense(y_train.shape[-1], activation="linear")(layers)
    model = Model(inputs, output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
                  loss=tf.keras.losses.MeanSquaredError(), metrics=["mae", "mse"])
    return model

def _reshape_inputs(da):
    arr = np.asarray(da.values)
    n_members, n_time, n_lat, n_lon = arr.shape
    return arr.reshape((n_members * n_time, n_lat, n_lon, 1)), n_members, n_time

def _reshape_labels(da):
    arr = np.asarray(da.values)
    if arr.ndim == 2: arr = arr[..., np.newaxis]
    return arr.reshape((arr.shape[0] * arr.shape[1], -1))

def load_presplit_data(settings):
    x_train_da = xr.open_dataarray(os.path.join(DATA_DIRECTORY, "X_train.nc"))
    x_val_da = xr.open_dataarray(os.path.join(DATA_DIRECTORY, "X_val.nc"))
    target = "rsut" if settings["exp_name"].endswith("rsut") else "rlut"
    y_train_da = xr.open_dataarray(os.path.join(DATA_DIRECTORY, f"y_train_{target}.nc"))
    y_val_da = xr.open_dataarray(os.path.join(DATA_DIRECTORY, f"y_val_{target}.nc"))

    x_train, _, _ = _reshape_inputs(x_train_da)
    x_val, _, _ = _reshape_inputs(x_val_da)
    y_train = _reshape_labels(y_train_da)
    y_val = _reshape_labels(y_val_da)
    return x_train, x_val, y_train, y_val

if __name__ == "__main__":
    EXP_NAMES = ["CanESM5_rsut", "CanESM5_rlut"]
    for name in EXP_NAMES:
        settings = get_settings(name)
        x_train, x_val, y_train, y_val = load_presplit_data(settings)
        
        model = compile_model(x_train, y_train, settings)
        print(f"Training {name} with GPU via ZLUDA...")
        
        model.fit(x_train, y_train, epochs=5, batch_size=settings["batch_size"], 
                  validation_data=(x_val, y_val), verbose=1)
        
        model.save(f"{MODEL_DIRECTORY}{name}_zluda_model.h5")
        print(f"Finished {name}")