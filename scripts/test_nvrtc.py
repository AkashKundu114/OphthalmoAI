import ctypes

cuda = ctypes.CDLL("libcuda.so.1")
nvrtc = ctypes.CDLL("libnvrtc.so.12")

cuda.cuInit(0)
dev = ctypes.c_int()
cuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

for arch in [b"sm_89", b"sm_90", b"sm_100", b"sm_120"]:
    kernel_code = b'extern "C" __global__ void test_k(float* x) { x[0] = 42.0f; }'
    prog = ctypes.c_void_p()
    nvrtc.nvrtcCreateProgram(ctypes.byref(prog), kernel_code, b"test.cu", 0, None, None)
    opts = [b"--gpu-architecture=" + arch]
    opts_arr = (ctypes.c_char_p * 1)(*opts)
    res = nvrtc.nvrtcCompileProgram(prog, 1, opts_arr)
    
    cubin_size = ctypes.c_size_t()
    nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_size))
    cubin_buf = (ctypes.c_char * cubin_size.value)()
    nvrtc.nvrtcGetCUBIN(prog, cubin_buf)
    
    mod = ctypes.c_void_p()
    load_err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin_buf)
    print(f"Arch {arch.decode()} CUBIN load err: {load_err}")
