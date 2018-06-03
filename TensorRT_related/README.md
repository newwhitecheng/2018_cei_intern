# Folder structure

1. The caffemodel and prototxt is at data/project folder, for example:
    ```Shell
    ./data/faster-rcnn/faster_rcnn_test_iplugin.prototxt    
    ./data/faster-rcnn/vgg16_faster_rcnn_iter_300000.caffemodel
    ```

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
    
# Performance
Faster-rcnn + VGG16 can process 20k images for 105mins (include resize).
    
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
    
# commands for dummies
1. copy trained model to tx2

    ```Shell
    scp /home/hc218/Downloads/tmp/py-faster-rcnn/output/faster_rcnn_end2end/val1/mobilenet_faster_rcnn_iter_10000.caffemodel \
    nvidia@10.236.176.251:/home/nvidia/workspace/TensorRT-3.0.4/data/faster-rcnn
    ```
 # TODO
 - [ ] Test Fater-rcnn + mobilenet performance
 - [ ] Test Faster-rcnn final with channel prunning (ICCV2017)
 
   
