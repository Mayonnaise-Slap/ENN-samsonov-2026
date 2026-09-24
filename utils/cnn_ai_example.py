from utils.model_calculations import CNN_calculate

layer = CNN_calculate(kernel=5, stride=1, padding=5 // 2, chan_in=32, chan_out=64)

ridge = 43
S = int(input("S: "))
B = int(input("B: "))

f = float(layer.flops(S, B))
b = float(layer.bytes_moved(S, B))
ai = f / b

verdict = "compute-bound" if ai > ridge else "memory-bound"
print(verdict)
