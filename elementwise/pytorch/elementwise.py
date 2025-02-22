import torch
from torch.utils.cpp_extension import load


def main():
    lib = load(
        name='elementwise_lib',  # name of the extension
        sources=['../elementwise.cu', "lib.cpp"],  # source files
        extra_include_paths=['../'],  # include paths
        extra_cflags=['-std=c++17'],  # extra flags
        verbose=True  # print the output of the compilation
    )
    a = torch.rand(1000, device='cuda')
    b = torch.rand(1000, device='cuda')
    c = torch.empty(1000, device='cuda')
    lib.ElementwiseAdd(a, b, c)
    for i in range(1000):
        assert c[i] == a[i] + b[i]
    for i in range(10):
        print(f'{a[i]} + {b[i]} = {c[i]}')
    print('Success!')


if __name__ == '__main__':
    main()
