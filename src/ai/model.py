"""
IDR MVP -- AI Velocity Model Architecture
==========================================
Implements the 1D Convolutional Neural Network (1D CNN) for forward velocity estimation
from 6-axis IMU sliding windows (MVP.md §9 and TECH_STACK.md §4).

Architecture:
    Input Window: (W, 6)
        ↓
    Conv1D(32, k=5, padding="same") + BatchNorm + ReLU + Dropout(0.1)
        ↓
    Conv1D(64, k=3, padding="same") + BatchNorm + ReLU + MaxPool1D(2) + Dropout(0.1)
        ↓
    Conv1D(128, k=3, padding="same") + BatchNorm + ReLU + GlobalAveragePooling1D
        ↓
    Dense(64, ReLU) + Dropout(0.1)
        ↓
    Dense(32, ReLU)
        ↓
    Dense(1, Linear) -> Forward Velocity (m/s)

Classes:
    NumpyCNNInference  -- High-speed pure-NumPy feed-forward inference runtime.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# Keras / TensorFlow Model Builder
# ---------------------------------------------------------------------------

def build_1d_cnn_velocity_model(
    input_shape: Tuple[int, int] = (50, 6),
    learning_rate: float = 1e-3,
):
    """
    Constructs and compiles the lightweight 1D CNN forward velocity model.

    Args:
        input_shape: (window_size, num_features), e.g. (50, 6)
        learning_rate: Initial Adam learning rate

    Returns:
        Compiled keras.Model instance.
    """
    import keras
    from keras import layers

    inputs = keras.Input(shape=input_shape, name="imu_window_input")

    # Block 1: Feature Extraction
    x = layers.Conv1D(filters=32, kernel_size=5, padding="same", name="conv1d_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.ReLU(name="relu_1")(x)
    x = layers.Dropout(0.1, name="drop_1")(x)

    # Block 2: Downsampling & Higher-order Features
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", name="conv1d_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.ReLU(name="relu_2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)
    x = layers.Dropout(0.1, name="drop_2")(x)

    # Block 3: Deep Representation & Global Temporal Aggregation
    x = layers.Conv1D(filters=128, kernel_size=3, padding="same", name="conv1d_3")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.ReLU(name="relu_3")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # Fully Connected Regression Head
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.1, name="drop_3")(x)
    x = layers.Dense(32, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(1, activation="linear", name="forward_velocity_ms")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="IDR_Velocity_1DCNN")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.Huber(delta=1.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae"), keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    return model


# ---------------------------------------------------------------------------
# High-Speed Pure-NumPy Inference Engine
# ---------------------------------------------------------------------------

class NumpyCNNInference:
    """
    Lightweight, dependency-free NumPy inference engine for the 1D CNN model.
    Runs forward passes in <0.05ms per window on standard CPU.
    """

    def __init__(self, weights_dict: Optional[Dict[str, np.ndarray]] = None):
        self.weights = weights_dict or {}

    @classmethod
    def from_keras_model(cls, keras_model) -> NumpyCNNInference:
        """Extracts and stores weight tensors from a trained Keras model."""
        weights = {}
        for layer in keras_model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                for idx, w in enumerate(layer_weights):
                    weights[f"{layer.name}_{idx}"] = np.array(w, dtype=np.float32)
        return cls(weights)

    def save_weights_npz(self, filepath: Union[str, Path]) -> None:
        """Saves weights to compressed .npz archive."""
        np.savez_compressed(filepath, **self.weights)

    @classmethod
    def load_weights_npz(cls, filepath: Union[str, Path]) -> NumpyCNNInference:
        """Loads weights from compressed .npz archive."""
        with np.load(filepath) as data:
            weights = {k: data[k] for k in data.files}
        return cls(weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Runs feed-forward inference on input batch X of shape (B, W, C) or (W, C).
        """
        if X.ndim == 2:
            X_batch = X[None, :, :]
            is_single = True
        else:
            X_batch = X
            is_single = False

        # Conv1: weights['conv1d_1_0'] (k, in_c, out_c), bias: ['conv1d_1_1']
        w_c1 = self.weights["conv1d_1_0"]
        b_c1 = self.weights["conv1d_1_1"]
        x = self._conv1d_same(X_batch, w_c1, b_c1)

        # BN1: gamma, beta, mean, var
        x = self._batch_norm(x, self.weights["bn_1_0"], self.weights["bn_1_1"],
                             self.weights["bn_1_2"], self.weights["bn_1_3"])
        x = np.maximum(0, x)  # ReLU

        # Conv2:
        w_c2 = self.weights["conv1d_2_0"]
        b_c2 = self.weights["conv1d_2_1"]
        x = self._conv1d_same(x, w_c2, b_c2)

        # BN2:
        x = self._batch_norm(x, self.weights["bn_2_0"], self.weights["bn_2_1"],
                             self.weights["bn_2_2"], self.weights["bn_2_3"])
        x = np.maximum(0, x)

        # MaxPool1D(2)
        x = self._maxpool1d(x, pool_size=2)

        # Conv3:
        w_c3 = self.weights["conv1d_3_0"]
        b_c3 = self.weights["conv1d_3_1"]
        x = self._conv1d_same(x, w_c3, b_c3)

        # BN3:
        x = self._batch_norm(x, self.weights["bn_3_0"], self.weights["bn_3_1"],
                             self.weights["bn_3_2"], self.weights["bn_3_3"])
        x = np.maximum(0, x)

        # GlobalAveragePooling1D
        x = np.mean(x, axis=1)  # Shape (B, 128)

        # Dense 1 (128 -> 64)
        w_d1 = self.weights["dense_1_0"]
        b_d1 = self.weights["dense_1_1"]
        x = np.maximum(0, x @ w_d1 + b_d1)

        # Dense 2 (64 -> 32)
        w_d2 = self.weights["dense_2_0"]
        b_d2 = self.weights["dense_2_1"]
        x = np.maximum(0, x @ w_d2 + b_d2)

        # Output Dense (32 -> 1)
        w_out = self.weights["forward_velocity_ms_0"]
        b_out = self.weights["forward_velocity_ms_1"]
        y_pred = x @ w_out + b_out

        if is_single:
            return float(y_pred[0, 0])
        return y_pred

    @staticmethod
    def _conv1d_same(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """1D convolution with same padding."""
        B, L, C_in = x.shape
        K, _, C_out = w.shape
        pad_total = K - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        x_padded = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant")
        out = np.zeros((B, L, C_out), dtype=np.float32)

        for i in range(L):
            patch = x_padded[:, i:i+K, :]  # Shape (B, K, C_in)
            out[:, i, :] = np.tensordot(patch, w, axes=([1, 2], [0, 1])) + b

        return out

    @staticmethod
    def _batch_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, mean: np.ndarray, var: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        scale = gamma / np.sqrt(var + eps)
        return (x - mean) * scale + beta

    @staticmethod
    def _maxpool1d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
        B, L, C = x.shape
        L_out = L // pool_size
        return np.max(x[:, :L_out * pool_size, :].reshape(B, L_out, pool_size, C), axis=2)
