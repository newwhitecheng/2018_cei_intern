# Troubleshooting
This file record the tensorRT (trt)errors and bugs that we've met. The purpose of this file is to provide a simple solution that worked for us and would probably work for you. 

1. error while loading shared libraries: libnvinfer.so.4: cannot open shared object file: No such file or directory

   **Note:** you might meet the above error. It's because you did not include the targets library to PATH.
    ```Shell
    # solution
    export LD_LIBRARY_PATH=/home/hc218/workspace/TensorRT-4.0.0.3/targets/x86_64-linux-gnu/lib:$PATH
    ```
   
2. ./bin/sample_fasterRCNN: error while loading shared libraries: libcudnn.so.7: cannot open shared object file: No such file or directory

   **Note:** you might meet the above error. It's because you did not include `cuda/lib64`.
    ```Shell
    # solution
    export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
    ```
   
