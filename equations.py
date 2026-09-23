import numpy as np

from utils.model_calculations import (
    CNN_calculate,
    GlobalAvgPool_calculate,
    Linear_calculate,
    MaxPool_calculate,
    ReLU_calculate,
)

LAYERS = [
    CNN_calculate(kernel=7, stride=2, padding=7 // 2, chan_in=3, chan_out=32),
    ReLU_calculate(channels=32),
    MaxPool_calculate(kernel=3, stride=2, padding=1, channels=32),
    CNN_calculate(kernel=5, stride=1, padding=5 // 2, chan_in=32, chan_out=64),
    ReLU_calculate(channels=64),
    CNN_calculate(kernel=3, stride=2, padding=3 // 2, chan_in=64, chan_out=128),
    ReLU_calculate(channels=128),
    CNN_calculate(kernel=1, stride=1, padding=1 // 2, chan_in=128, chan_out=256),
    ReLU_calculate(channels=256),
    CNN_calculate(kernel=3, stride=2, padding=3 // 2, chan_in=256, chan_out=256),
    ReLU_calculate(channels=256),
    CNN_calculate(kernel=1, stride=1, padding=1 // 2, chan_in=256, chan_out=512),
    ReLU_calculate(channels=512),
    GlobalAvgPool_calculate(channels=512),
    Linear_calculate(in_features=512, out_features=256),
    ReLU_calculate(channels=256),
    Linear_calculate(in_features=256, out_features=100),
]


def _forward_sizes(image_size: np.ndarray):
    size = np.asarray(image_size)
    for layer in LAYERS:
        yield layer, size
        size = layer.output_size(size)  # catch and update at the end of layer iter


def flops(image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
    image_size = np.asarray(image_size)
    batch = np.asarray(batch)
    total = 0.0
    for layer, size in _forward_sizes(image_size):
        total += layer.flops(size, batch)
    return total


def bytes_moved(image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
    image_size = np.asarray(image_size)
    batch = np.asarray(batch)
    total = 0.0
    for layer, size in _forward_sizes(image_size):
        total = total + layer.bytes_moved(size, batch)
    return total


def memory(image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:  # bytes
    image_size = np.asarray(image_size)
    batch = np.asarray(batch)
    weight_bytes = sum(layer.weight_bytes() for layer in LAYERS)
    activation_peaks = [
        layer.peak_activation_bytes(size, batch) for layer, size in _forward_sizes(image_size)
    ]
    return weight_bytes + np.maximum.reduce(activation_peaks)


def latency(image_size: np.ndarray, batch: np.ndarray, theta) -> np.ndarray:  # seconds
    f = flops(image_size, batch)
    m = bytes_moved(image_size, batch)
    compute_time = f / theta["flops_rate"]
    memory_time = m / theta["bandwidth"]
    return theta["t_startup"] + np.maximum(compute_time, memory_time)


def energy(image_size: np.ndarray, batch: np.ndarray, theta_energy) -> np.ndarray:  # joules
    f = flops(image_size, batch)
    m = bytes_moved(image_size, batch)
    t = latency(image_size, batch, theta_energy["latency"])
    return (
            theta_energy["p_idle"] * t
            + theta_energy["e_flop"] * f
            + theta_energy["e_byte"] * m
    )
