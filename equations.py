import numpy as np

from models import CNN_params

# FP32
Q_bytes = 4

_cnn_S_out_memo = {}
# CNN

# memorize? hardcode?
def _cnn_S_out(image_size: np.ndarray, cnn: CNN_params) -> np.ndarray:
    return np.ceil((image_size + 2 * cnn.padding - cnn.kernel) / cnn.stride)


def _cnn_flops(image_size: np.ndarray, batch: np.ndarray, cnn: CNN_params) -> np.ndarray:
    return 2 * batch * _cnn_S_out(image_size, cnn) ** 2 * cnn.kernel ** 2 * cnn.chan_in * cnn.chan_out


def _cnn_memory(image_size: np.ndarray, batch: np.ndarray, cnn: CNN_params) -> np.ndarray:
    return (
            cnn.kernel ** 2 * cnn.chan_in * cnn.chan_out +
            image_size ** 2 *  cnn.chan_in * batch +
            _cnn_S_out(image_size, cnn) * cnn.chan_out * batch
    ) * Q_bytes


# MaxPool
def _maxpool_flops(image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
    pass


def _maxpool_memory(image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
    pass


# GlobalAvgPool
def _global_flops(image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
    pass


def _global_memory(image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
    pass


# Linear
def _linear_flops(image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
    pass


def _linear_memory(image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
    pass


# overall
def flops(image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
    pass


def memory(image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:  # bytes
    pass


def latency(image_size: np.ndarray, batch: np.ndarray, theta) -> np.ndarray:  # seconds
    pass


def energy(image_size: np.ndarray, batch: np.ndarray, theta_energy) -> np.ndarray:  # joules
    pass
