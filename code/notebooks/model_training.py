import gc
import os
import random
import sys
import pickle
import json
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
import xarray as xr
from keras import Model
from keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D

# Setup Directories
MODEL_DIRECTORY = "../../data/saved_models/"
PREDICTIONS_DIRECTORY = "../../data/saved_predictions/"
DATA_DIRECTORY = "../../data/splits"
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
            "hiddens": [32, 16], "act_fun": ["elu", "elu"], "learning_rate": 0.00005,
            "batch_size": 32, "rng_seed": 42, "rng_seed_list": np.arange(5).tolist(), "n_epochs": 5_000,
        },
        "CanESM5_rlut": {
            "input_region": "X_train.nc", "detrend": True, "datafolder": "../../data/splits/",
            "anomalies": True, "anomalies_years": (0, 30), "data_period":"_1850-2100",
            "input_var": "tas", "label_var": "cre", "n_train_val_test": (17,4,4),
            "yr_bounds": (1850, 2014), "network_type": "cnn", "kernel_size": 3,
            "kernels": [32, 32], "ridge_param": [0.0, 0.0], "kernel_act": ["relu", "relu"],
            "hiddens": [32, 16], "act_fun": ["elu", "elu"], "learning_rate": 0.00005,
            "batch_size": 32, "rng_seed": 42, "rng_seed_list": np.arange(5).tolist(), "n_epochs": 5_000,
        },
    }
    exp_dict = experiments[experiment_name]
    exp_dict['exp_name'] = experiment_name
    if 'n_train_val_test' in exp_dict.keys():
        exp_dict['n_members'] = int(np.sum(exp_dict["n_train_val_test"]))
    return exp_dict


def plot_metrics(history, metric):
    imin = np.argmin(history.history["val_loss"])

    plt.plot(history.history[metric], label="training")
    plt.plot(history.history["val_" + metric], label="validation")
    plt.title(metric)
    plt.axvline(x=imin, linewidth=0.5, color="gray", alpha=0.5)
    plt.legend()


def plot_metrics_panels(history, predictions):
    baseline_mae = np.mean(np.abs(predictions["labels_test"] - 0.0))
    baseline_mse = np.mean((predictions["labels_test"] - 0.0) ** 2)

    plt.subplots(figsize=(20, 4))

    plt.subplot(1, 4, 1)
    plot_metrics(history, "loss")
    plt.ylim(0, 0.5)
    plt.axhline(y=baseline_mse, color="gray", linestyle="--", label="baseline")

    plt.subplot(1, 4, 2)
    plot_metrics(history, "mae")
    plt.ylim(0, 0.5)
    plt.axhline(y=baseline_mae, color="gray", linestyle="--", label="baseline")

    plt.legend()


def plot_pred_vs_truth(predictions, settings, ms=5, label=""):
    mse = sklearn.metrics.mean_squared_error(predictions["labels_val"], predictions["pred_val"])

    if np.isnan(mse):
        mse = 0.0
    plt.plot(
        predictions["labels_val"],
        predictions["pred_val"],
        ".",
        alpha=0.5,
        label=label + " (MSE=" + str(np.round(mse, 3)) + ")",
        markersize=ms,
    )

    plt.plot((-10, 10), (-10, 10), "--k", alpha=0.5)
    plt.ylim(-8, 3)
    plt.xlim(-8, 3)
    plt.ylabel("predicted")
    plt.xlabel("truth")
    plt.legend(frameon=False, fontsize=8)
    plt.gca().set_aspect("equal")


def compile_model(x_train, y_train, settings):
    inputs = Input(shape=x_train.shape[1:])
    layers = inputs

    if settings["network_type"] == "ffn":
        assert len(settings["hiddens"]) == len(settings["act_fun"]) == len(settings["ridge_param"])

        layers = Flatten()(layers)
        layers = Dropout(rate=settings["dropout_rate"][0], seed=settings["rng_seed"])(layers)

        for hidden, activation, ridge in zip(settings["hiddens"], settings["act_fun"], settings["ridge_param"]):
            layers = Dense(
                hidden,
                activation=activation,
                use_bias=True,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=ridge),
                bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
            )(layers)

    elif settings["network_type"] == "cnn":
        assert len(settings["kernels"]) == len(settings["kernel_act"])
        assert len(settings["hiddens"]) == len(settings["act_fun"])

        for kernel, kernel_act in zip(settings["kernels"], settings["kernel_act"]):
            layers = Conv2D(kernel, (settings["kernel_size"], settings["kernel_size"]),
                            use_bias=True,
                            activation=kernel_act,
                            padding="same",
                            bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
                            kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
                            )(layers)
            layers = MaxPooling2D((2, 2))(layers)

        layers = Flatten()(layers)
        for hidden, activation in zip(settings["hiddens"], settings["act_fun"]):
            layers = Dense(
                hidden,
                activation=activation,
                use_bias=True,
                bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
            )(layers)

    else:
        raise NotImplementedError()

    output_layer = Dense(
        y_train.shape[-1],
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
    )(layers)
    model = Model(inputs, output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mae", "mse"],
    )
    return model


def get_model_name(settings, suffix=""):
    model_name = (settings["exp_name"] + suffix + '_rngseed' + str(settings["rng_seed"]))
    return model_name


def save_predictions(predictions, filename):
    with open(filename, 'wb') as f:
        pickle.dump(predictions, f)


def save_tf_model(model, model_name, directory, settings):
    tf.keras.models.save_model(model, directory + model_name + "_model.keras", overwrite=True)

    with open(directory + model_name + '_metadata.json', 'w') as json_file:
        json_file.write(json.dumps(settings))


def _open_dataarray(filepath):
    try:
        return xr.open_dataarray(filepath)
    except Exception:
        ds = xr.open_dataset(filepath)
        first_var = list(ds.data_vars)[0]
        return ds[first_var]


def _reshape_inputs(da):
    arr = np.asarray(da.values)
    if arr.ndim != 4:
        raise ValueError(f"Expected input with 4 dims (member, time, lat, lon), got shape {arr.shape}")

    n_members, n_time, n_lat, n_lon = arr.shape
    x = arr.reshape((n_members * n_time, n_lat, n_lon, 1))
    return x, n_members, n_time, (n_lat, n_lon)


def _reshape_labels(da):
    arr = np.asarray(da.values)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    elif arr.ndim < 2:
        raise ValueError(f"Expected label with at least 2 dims (member, time, ...), got shape {arr.shape}")

    y = arr.reshape((arr.shape[0] * arr.shape[1], -1))
    return y


def load_presplit_data(settings, split_dir=None, verbose=1):
    if split_dir is None:
        split_dir = DATA_DIRECTORY
    split_dir = os.path.normpath(split_dir)

    target_suffix = "rsut" if settings["exp_name"].endswith("rsut") else "rlut"

    x_train_da = _open_dataarray(os.path.join(split_dir, "X_train.nc"))
    x_val_da = _open_dataarray(os.path.join(split_dir, "X_val.nc"))
    x_test_da = _open_dataarray(os.path.join(split_dir, "X_test.nc"))

    y_train_da = _open_dataarray(os.path.join(split_dir, f"y_train_{target_suffix}.nc"))
    y_val_da = _open_dataarray(os.path.join(split_dir, f"y_val_{target_suffix}.nc"))
    y_test_da = _open_dataarray(os.path.join(split_dir, f"y_test_{target_suffix}.nc"))

    x_train, train_members, train_time, map_shape = _reshape_inputs(x_train_da)
    x_val, val_members, _, _ = _reshape_inputs(x_val_da)
    x_test, test_members, _, _ = _reshape_inputs(x_test_da)

    y_train = _reshape_labels(y_train_da)
    y_val = _reshape_labels(y_val_da)
    y_test = _reshape_labels(y_test_da)

    lat = x_train_da["lat"].values if "lat" in x_train_da.coords else np.arange(map_shape[0])
    lon = x_train_da["lon"].values if "lon" in x_train_da.coords else np.arange(map_shape[1])

    if verbose == 1:
        print("Loaded pre-split data:")
        print(f"  X_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val:   {x_val.shape}, y_val:   {y_val.shape}")
        print(f"  X_test:  {x_test.shape}, y_test:  {y_test.shape}")

    norm_dict = {}
    member_enrollment = (
        np.arange(train_members),
        np.arange(val_members),
        np.arange(test_members),
    )

    member_shape = train_members
    time_shape = train_time

    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        lat,
        lon,
        map_shape,
        member_shape,
        time_shape,
        norm_dict,
        member_enrollment,
        settings,
    )


if __name__ == "__main__":
    plt.style.use("default")
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["figure.dpi"] = 150
    savefig_dpi = 300

    print(f"python version = {sys.version}")
    print(f"numpy version = {np.__version__}")
    print(f"xarray version = {xr.__version__}")
    print(f"tensorflow version = {tf.__version__}")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    exp_name_list = (
        "CanESM5_rsut",
        "CanESM5_rlut",
    )
    overwrite_model = False

    for exp_name in exp_name_list:
        print("------" + exp_name + "------")
        settings = get_settings(exp_name)

        for rng_seed in settings["rng_seed_list"]:
            settings["rng_seed"] = rng_seed
            tf.random.set_seed(settings["rng_seed"])
            random.seed(settings["rng_seed"])
            np.random.seed(settings["rng_seed"])

            model_name = get_model_name(settings)
            if os.path.exists(MODEL_DIRECTORY + model_name + "_model.keras") and overwrite_model is False:
                print("\n" + model_name + " exists. Skipping...")
                print("================================\n")
                continue

            (
                x_train,
                x_val,
                x_test,
                labels_train,
                labels_val,
                labels_test,
                lat,
                lon,
                map_shape,
                member_shape,
                time_shape,
                norm_dict,
                member_enrollment,
                settings,
            ) = load_presplit_data(settings, split_dir=DATA_DIRECTORY, verbose=1)

            tf.keras.backend.clear_session()
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, verbose=1, mode="auto", restore_best_weights=True
            )

            model = compile_model(x_train, labels_train, settings)
            model.summary()

            history = model.fit(
                x_train,
                labels_train,
                epochs=settings["n_epochs"],
                batch_size=settings["batch_size"],
                shuffle=True,
                validation_data=[x_val, labels_val],
                callbacks=[early_stopping],
                verbose=2,
            )

            predictions = {
                "labels_train": labels_train,
                "pred_train": model.predict(x_train),
                "labels_val": labels_val,
                "pred_val": model.predict(x_val),
                "labels_test": labels_test,
                "pred_test": model.predict(x_test),
            }

            _ = gc.collect()

            save_tf_model(model, model_name, MODEL_DIRECTORY, settings)
            save_predictions(predictions, PREDICTIONS_DIRECTORY + model_name + "_predictions.pickle")

            plot_metrics_panels(history, predictions)
            plt.savefig(
                DIAGNOSTICS_DIRECTORY + model_name + "_metrics_diagnostic.png",
                dpi=savefig_dpi,
                bbox_inches="tight",
            )
            plt.close()

            plot_pred_vs_truth(predictions, settings)
            plt.savefig(
                FIGURES_DIRECTORY + model_name + "_pred_vs_truth.png",
                dpi=savefig_dpi,
                bbox_inches="tight",
            )
            plt.close()

            _ = gc.collect()

        print("------" + exp_name + " DONE ------")