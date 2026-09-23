class CNN_params:
    def __init__(self, kernel: int, stride: int, padding: int, chan_in: int, chan_out: int):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.chan_in = chan_in
        self.chan_out = chan_out
