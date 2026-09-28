import ctypes
import os
import subprocess

print("--- Testing CUDA Driver & Cubin Loading ---")
cuda = ctypes.CDLL("libcuda.so.1")
res = cuda.cuInit(0)
print(f"cuInit: {res}")

dev = ctypes.c_int()
cuda.cuDeviceGet(ctypes.byref(dev), 0)
name = (ctypes.c_char * 64)()
cuda.cuDeviceGetName(name, 64, dev)
print(f"Device name: {name.value.decode()}")

ctx = ctypes.c_void_p()
cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

# Extract a cubin from libtorch_cuda.so
out_cubin = "/tmp/test_120.cubin"
cuobjdump = "/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/cuobjdump"
libtorch = "/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so"

ret = subprocess.run([cuobjdump, "-xelf", "libtorch_cuda.1204.sm_120.cubin", libtorch], cwd="/tmp", capture_output=True, text=True)
print(f"cuobjdump ret: {ret.returncode}, stdout: {ret.stdout.strip()[:100]}, stderr: {ret.stderr.strip()[:100]}")

extracted = "/tmp/libtorch_cuda.1204.sm_120.cubin"
if os.path.exists(extracted):
    mod = ctypes.c_void_p()
    load_err = cuda.cuModuleLoad(ctypes.byref(mod), extracted.encode())
    print(f"cuModuleLoad of extracted sm_120 cubin error: {load_err}")
    if load_err != 0:
        # Check error string
        err_str = ctypes.c_char_p()
        cuda.cuGetErrorString(load_err, ctypes.byref(err_str))
        print(f"Error string: {err_str.value.decode() if err_str.value else 'unknown'}")
else:
    print(f"File {extracted} not found!")
