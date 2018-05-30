# How to run
The following instruction shows you how to run our submission executable file.
1. log in to one of the tx2
    ```Shell
    ssh nvidia@10.236.176.251
    ```
2. change directory and run
    ```Shell
    cd /home/nvidia/workspace/TensorRT-3.0.4/bin
    ./sample_fasterRCNN
    ```
3. The result will be named submission.csv at the same directory
    ```Shell
    ls 
    ```
    you will see submission.csv

# Troubleshooting
This file record the tensorRT (trt)errors and bugs that we've met. The purpose of this file is to provide a simple solution that worked for us and would probably work for you. 

1. error while loading shared libraries: libnvinfer.so.4: cannot open shared object file: No such file or directory

   **Note:** you might meet the above error. It's because you did not include the targets library to PATH.
    ```Shell
    # solution: export the target library
    # for example 
    export LD_LIBRARY_PATH=/home/hc218/workspace/TensorRT-4.0.0.3/targets/x86_64-linux-gnu/lib:$PATH
    ```
   
2. ./bin/sample_fasterRCNN: error while loading shared libraries: libcudnn.so.7: cannot open shared object file: No such file or directory

   **Note:** you might meet the above error. It's because you did not include `cuda/lib64`.
    ```Shell
    # solution: export the cuda lib64 path
    # for example:
    export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
    ```
   
