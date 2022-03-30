from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_align',
    ext_modules=[
        Extension('roi_align_cuda', 
        [
            'src/roi_align.cu',
            'src/roi_align_api.cpp', 
            'src/roi_align_pybind.cpp'
        ],
        include_dirs=['/home/fcq/Li/Mess/cuda_pytorch_programme/src/', '/home/fcq/anaconda3/envs/multimodal/lib/python3.8/site-packages/torch/include', '/home/fcq/anaconda3/envs/multimodal/lib/python3.8/site-packages/torch/include/torch/csrc/api/include', '/home/fcq/anaconda3/envs/multimodal/lib/python3.8/site-packages/torch/include/TH', '/home/fcq/anaconda3/envs/multimodal/lib/python3.8/site-packages/torch/include/THC', '/usr/local/cuda-11.0/include'],
        library_dirs=['/home/fcq/anaconda3/envs/multimodal/lib/python3.8/site-packages/torch/lib', '/usr/local/cuda-11.0/lib64'],
        libraries=['c10', 'torch', 'torch_cpu', 'torch_python', 'cudart', 'c10_cuda', 'torch_cuda_cu', 'torch_cuda_cpp'],
        extra_compile_args={'cxx': ['-std=c++14'], 'nvcc': ['-std=c++14']},
        language='c++',
                            )
    ],
    cmdclass={'build_ext': BuildExtension}
)

