# Yolov8 Aim Assist with IFF
## Disclaimer 
- This is a modified branch of project: [Franklin-Zhang0/Yolo-v8-Apex-Aim-assist](https://github.com/Franklin-Zhang0/Yolo-v8-Apex-Aim-assist). Kudos to Franklin! If you find this project useful, please consider starring the original repo!
- This project has no intend of being utilized in game play and it should not be. Use at your own risk.
- This project will not be regularly maintained as it already meets the goal(for now). Update might be provided once better models/compression methods/quantization methods are available.

P.S. This adds a working environment.yml file and an updated README setup instructions since the previous repositories had no versions in the requirements.txt.

| Model        | Size<br><sup>(pixels) | Enemy prediction<sup>val<br>mAP 50-95 | Teammate prediction<sup>val<br>mAP 50-95 | Speed<br><sup>.trt fp16<br>(fps) | Speed<br><sup>.trt int8<br>(fps) | IFF performence |
| -------------| --------------------- | -------------------- | -------------------- | ------------------------------ | ------------------------------- | ------------------ |
| Apex_8n      | 640                   | 54.5                 | 71.3                 | 58.88                          | 61.23                           | Poor                |
| Apex_8s      | 640                   | 66.9                 | 79.2                 | 27.10                          | 33.43                           | Acceptable               |
| Apex_8n_pruned| 640                  | 44.1                 | 55.9                 | 72.22                          | 74.04                           | Poor               |
| Apex_8s_pruned| 640                  | 52.9                 | 64.9                 | 30.54                          | 31.82                           | Poor               |


## Advantages and Todos
Advantages:
* [x] Fast screencapturing with dxshot
* [x] Not Logitech Macro/GHub dependent
* [x] Customizable trigger keys
* [x] PID capable
* [x] IFF(Identification friend or foe) capable (Works 50/50)
* [x] fp16 precision

Todos:
* [x] int8 precision (No significant improvement as for now)
* [x] Pruning (Sacrificies accuracy way too much)
* [ ] Increase accuracy of IFF
* [ ] Increase accuracy under: ADS, partial body exposure, gunfire blockage, smoke...

## 1. Set up the environment

- Version-align

    |  CUDA   |  cuDNN   | TensorRT | PyTorch  |
    | :-----: | :------: | :------: | :------: |
    | 12.1.1  | 8.9.0    |  8.6.1.1 | 2.2.1    |


### 1.1. Environment set up under Linux
- Install `Conda` (if not already installed)

    In your terminal window, run:`bash Miniconda3-latest-Linux-x86_64.sh` to install `Miniconda` (suggested) or run:`bash Anaconda-latest-Linux-x86_64.sh` to install `Anaconda`.
    
- Setup a new environment
    
    In your terminal window, run:
    ```shell
    conda create -n yolov8 python=3.10 # create environment 'yolov8' with python 3.10 installed 
    conda activate yolov8 # activate environment 'yolov8'
    ```
    
- Install `CUDA` and `PyTorch`.
   
   ~~if you have a cuda capable gpu, you can running the following extra command~~ 

  Running inferrence on CPU is suboptimal and no longer suggested by this project. Please use `CUDA` capable GPU and run the following command to install `CUDA`, `cuDNN` and `PyTorch`:
   ``` shell
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`
   ```
- Install `TensorRT`.
 
  Add the following command
   ```
   pip install --upgrade setuptools pip --user
   pip install nvidia-pyindex
   pip install --upgrade nvidia-tensorrt
   pip install pycuda
   ```
 - Install python requirement.
   ``` shell
   pip install -r requirement.txt
   ```
    

### 1.2. Environment set up in Windows10 & Windows11

- In your terminal window, run:
    ```shell
    conda create -n yolov8 python=3.10 # create environment 'yolov8' with python 3.10 installed 
    conda activate yolov8 # activate environment 'yolov8'
    ```

- Install `CUDA`. (One can also follow the official instruction:[`CUDA official website`](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)).
DOWNLOAD NVIDIA CUDA COMPUTING TOOLKIT 12.1.1 AND RUN THE BELOW COMMAND
    ```shell
    conda install cuda -c nvidia/label/cuda-12.1.1 # install CUDA 12.1.1
    ```

- Install `cuDNN`.
    - Register for the [`NVIDIA developer program`](https://developer.nvidia.com/login).
    - Go to the cuDNN download site:[`cuDNN download archive`](https://developer.nvidia.com/rdp/cudnn-archive).
     - Click `Download cuDNN v8.9.0 (April 11th, 2023), for CUDA 12.x`.
    - Unzip it
    - Copy all three folders (`bin`,`include`,`lib`) and paste them to the `CUDA` installation directory `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`. (NOTE `bin`,`include`,`lib` folders are already exist in the CUDA folder.).

- Install `PyTorch`.
For reliability, you need to download the CUDA versions of pytorch, torchvision and torchaudio from https://download.pytorch.org/whl/cu121
Look for the versions with cu121 and cp10 win64 and pip install them
    ```shell
    conda install pytorch-cuda=12.1.1 -c pytorch -c nvidia
    ```
- Install `TensorRT`.
    Follow the [Nvidia instruction of installation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip).
    - Go to the [TensorRT download site](https://developer.nvidia.com/nvidia-tensorrt-8x-download).
    - Download `TensorRT 8.6 GA for Windows 10 and CUDA 12.0 and 12.1 ZIP Package`.
    - Unzip the folder
    - Add the `<your install path>\TensorRT-8.6.1.6\lib` into PATH environment variable.
    - Go to the folder `<your install path>\TensorRT-8.6.1.6\python`
    - In command window, input 
        ```shell
        conda activate yolov8 # activate dedicated environment
        pip install tensorrt-8.6.1.6-cp310-none-win_amd64.whl # install tensorrt package to python
        ```
 - Install python requirement.
   ``` shell
   pip install -r requirements.txt
   ```

<details>
<summary> Verify installation and check versions.</summary>
    
- Verify installation of `CUDA`, `cuDNN`, `PyTorch` and `TensorRT`.  
    
    - Verify `CUDA`.
        ```shell
        nvcc -V
        ```
        If installment successed, you should see prompts like:
        ```shell
        nvcc: NVIDIA (R) Cuda compiler driver
        Copyright (c) 2005-2022 NVIDIA Corporation
        Built on Tue_May__3_19:00:59_Pacific_Daylight_Time_2022
        Cuda compilation tools, release 11.7, V11.7.64
        Build cuda_11.7.r11.7/compiler.31294372_0
        ```
     - Verify `cuDNN`.
     
        ```shell
        python
        import torch
        print(torch.backends.cudnn.version())
        ```
        
     - Verify `PyTorch`.
      
        ```shell
        python
        import torch
        print(torch.__version__)
        ```
    
    - Verify `TensorRT`.
      
        ```shell
        pip show tensorrt
        ```
        If installment successed, you should see prompts like:
        ```shell
        Name: tensorrt
        Version: 8.5.2.2
        Summary: A high performance deep learning inference library
        Home-page: https://developer.nvidia.com/tensorrt
        Author: NVIDIA Corporation
        ```    
    
    
    
</details>



## 2. Build your weight
### 2.1. PyTorch `.pt` weight
You have several options here to realize your `.pt` weight:
- (1) Use the provided weight `apex_v8n.pt`.
     
     This is a weight based on `yolov8n.pt`, trained with 7W screenshots and labeled 'Enemy', 'Teammate'. Due to the nature of this project, the provided weight is poorly trained to prevent abuse and cheating. However, it can already track characters well and demonstrate some level of IFF capability.  
     
- (2) Use the provided weight `apex_v8n.pt` as a pretrained weight and train your own weight with your own dataset.
     
     Please follow the [official instruction](https://docs.ultralytics.com/usage/cli/) of `Ultralytics` to train your own weight.
     
     Note that the dataset is required to use `YOLO` annotation format, please reform your dataset into the following structure:
        
    ```shell
    dataset/apex/
    ├── train
    |   ├── images
    │   |   ├── 000000000001.jpg
    │   |   ├── 000000580008.jpg
    │   |   |   ...
    |   ├── lables    
    │   |   ├── 000000000001.txt
    │   |   ├── 000000580008.txt
    │   |   |   ...
    ├── valid
    |   ├── images
    │   |   ├── 000000000113.jpg
    │   |   ├── 000000000567.jpg
    |   ├── lables    
    │   |   ├── 000000000113.txt
    │   |   ├── 000000000567.txt
    │   |   |   ...    
    ```
    
- (3) Train your own weight with official pretrained `yolov8` weights.
    
    If the provided weight which is based on `yolov8n.pt` can not meet your expectation. You can also explore the options of other pretrained weights provided by `yolov8`.
    
    - Model speed: `8n>8s>8m`
    
    - Model accuracy: `8n<8s<8m`
    
    Please follow the [official instruction](https://docs.ultralytics.com/usage/cli/) of `Ultralytics` to train your own weight.
    
### 2.2. ONNX `.onnx` weight (skip if only fp16 precision is desired)

You have several options here to to convert your `.pt` weight to a `.onnx` weight.

- (1) Use yolov8 built in function `YOLO export`:

    ```shell
    yolo export model=<your weight path>/best.pt format=onnx
    ```
    Note this built-in method is identical to the python code provided in [TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)
    
- (2) Use Paddleslim ACT (In Linux):
    
    

### 2.3. TensorRT `.trt` or `.engine` weight

Use any of the following methods to generate TensorRT engine:

- (1) Use yolov8 built-in function `YOLO export` to export `.engine` weight directly from `.pt` weight.
    
    ```shell
    # out put fp32 precision (default)
    yolo export model=<your weight path>/best.pt format=engine
    
    # out put fp16 precision (recommanded)
    yolo export model=<your weight path>/best.pt format=engine fp16=True
    ```
- (2) Use the third-party method [TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series) to export `.trt` weight from previous generated `.onnx` weight.

    ```shell
    # out put fp32 precision
    python export.py -o <your weight path>/best.onnx -e apex_fp32.trt -p fp32 --end2end --v8
    
    # out put fp16 precision (default, recommanded)
    python export.py -o <your weight path>/best.onnx -e apex_fp16.trt --end2end --v8
    
    # out put int8 precision (for extreme performance)
    python export.py -o <your weight path>/best.onnx -e apex_int8.trt -p int8 --calib_input <your data path>/train/images --calib_num_images 500 --end2end --v8
    ```
    
## 3. Run the program
Replace the model and change settings in the `args_.py` file (See Section 4.)

Run the `main.py` file with the following command
```shell        
conda activate v8trt # activate dedeicated env
python main.py # start aim assist
```
After a few seconds, the program will start fucntioning. You should see the follwing prompts in the console:
```shell
listener start
Main start
Screenshot fps:  311.25682541048934
fps:  61.09998811302416
interval:  0.016366615295410156
```

Explaination of keys:
- `Shift`: Holding shift triggers aim assist. By default, only `holding shift` can trigger the aim assist.
- `Left`: Unlock `left mouse button`. Once `left` key is clicked, you should hear a beep and now holding `left mouse button` can also trigger aim assist 
- `Right`: Unlock `right mouse button`. Once `right` key is clicked, you should hear a beep and now holding `right mouse button` can also trigger aim assist 
- `End`: Cilck `End` for continious aiming. Auto aiming is always on and another click to turn off.
- `Left mouse button`: Optional trigger
- `Right mouse button`: Optional trigger
- `Home`: Stop listening and exit program

## 4. Change settings
You can change following settings in the `args_.py` file.
- `--model`: The weight to be used by this project. Please replace this with your own `.trt` or `.engine` weight.
- `--classes`: classes to be detected, can be expanded but need to be an array. For example, 0 represents 'Teammate', 1 represents 'Enemy'. Then the input should be [1].
- `--conf`: Confidence level for inference. Adjust it based on your model accuracy.
- `--crop_size`: The portion to detect from the screen. Adjust it based on your monitor resolution. For example: `1/3` for 1440P, or `1/2` for 1080P.
- `--pid`: Use PID controller to smooth aiming and prevent overdrifting. Leave it by default is recommanded.
- `--Kp`,`--Ki`,`--Kd`: PID components that need to be carefully calibrated. `--Kp=0.5`, `--Ki=0.1`, `--Kd=0.1` is recommanded as starting point.
You can also modify the `MyListener.py` file.
- Function `def listen_key(key)` and def `keys_released(key)`: Change `keyboard.Key.home`, `keyboard.Key.end`, `keyboard.Key.shift`, `keyboard.Key.left` or `keyboard.Key.right` to whatever keys you like to customize the key settings.

## Reference
[Train image dataset](https://universe.roboflow.com/apex-esoic/apexyolov6)

[TensorRt code](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)
