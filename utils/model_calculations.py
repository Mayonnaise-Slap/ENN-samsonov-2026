from abc import ABC, abstractmethod

import numpy as np

# FP32
Q_bytes = 4


class LayerCalculate(ABC):
    @abstractmethod
    def output_size(self, image_size: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def flops(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def weight_bytes(self) -> float:
        pass

    @abstractmethod
    def activation_in_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def activation_out_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        pass

    def bytes_moved(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return (
                self.activation_in_bytes(image_size, batch)
                + self.activation_out_bytes(image_size, batch)
                + self.weight_bytes()
        )

    def peak_activation_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return self.activation_in_bytes(image_size, batch) + self.activation_out_bytes(image_size, batch)


class CNN_calculate(LayerCalculate):
    def __init__(self, kernel: int, stride: int, padding: int, chan_in: int, chan_out: int):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.chan_in = chan_in
        self.chan_out = chan_out

    def output_size(self, image_size: np.ndarray) -> np.ndarray:
        return np.floor((image_size + 2 * self.padding - self.kernel) / self.stride) + 1

    def flops(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        s_out = self.output_size(image_size)
        return 2 * batch * s_out ** 2 * self.kernel ** 2 * self.chan_in * self.chan_out

    def weight_bytes(self) -> float:
        return self.kernel ** 2 * self.chan_in * self.chan_out * Q_bytes

    def activation_in_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return image_size ** 2 * self.chan_in * batch * Q_bytes

    def activation_out_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return self.output_size(image_size) ** 2 * self.chan_out * batch * Q_bytes


class MaxPool_calculate(LayerCalculate):
    def __init__(self, kernel: int, stride: int, padding: int, channels: int):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.channels = channels

    def output_size(self, image_size: np.ndarray) -> np.ndarray:
        return np.floor((image_size + 2 * self.padding - self.kernel) / self.stride) + 1

    def flops(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return 0.0 * batch * self.output_size(image_size)

    def weight_bytes(self) -> float:
        return 0.0

    def activation_in_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return image_size ** 2 * self.channels * batch * Q_bytes

    def activation_out_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return self.output_size(image_size) ** 2 * self.channels * batch * Q_bytes


class ReLU_calculate(LayerCalculate):
    def __init__(self, channels: int):
        self.channels = channels

    def output_size(self, image_size: np.ndarray) -> np.ndarray:
        return image_size

    def flops(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return 0.0 * batch * image_size

    def weight_bytes(self) -> float:
        return 0.0

    def activation_in_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return image_size ** 2 * self.channels * batch * Q_bytes

    def activation_out_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return self.activation_in_bytes(image_size, batch)

    def peak_activation_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return self.activation_in_bytes(image_size, batch)


class GlobalAvgPool_calculate(LayerCalculate):
    def __init__(self, channels: int):
        self.channels = channels

    def output_size(self, image_size: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(image_size, dtype=float))

    def flops(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return 0.0 * batch * image_size

    def weight_bytes(self) -> float:
        return 0.0

    def activation_in_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return image_size ** 2 * self.channels * batch * Q_bytes

    def activation_out_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return self.channels * batch * Q_bytes * np.ones_like(np.asarray(image_size, dtype=float))


class Linear_calculate(LayerCalculate):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def output_size(self, image_size: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(image_size, dtype=float))

    def flops(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return 2 * batch * self.in_features * self.out_features * np.ones_like(np.asarray(image_size, dtype=float))

    def weight_bytes(self) -> float:
        return (self.in_features * self.out_features + (self.out_features if self.bias else 0)) * Q_bytes

    def activation_in_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return self.in_features * batch * Q_bytes * np.ones_like(np.asarray(image_size, dtype=float))

    def activation_out_bytes(self, image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
        return self.out_features * batch * Q_bytes * np.ones_like(np.asarray(image_size, dtype=float))
