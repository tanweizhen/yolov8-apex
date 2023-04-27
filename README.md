# Yolo v8 Aim Assist
## If you like it, a star is appreciated!!!

## 1. How to set up the environment

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

The following method has being tested and successed under `Windows 10 Pro Version 21H2/22H2`, `Windows11 Pro Version 22H2` and `Windows11 Pro Insider View Build 25346`. Technically, it works under all latest `Windows OS` builds.
- Version-align

    |  CUDA   |  cuDNN   | TensorRT | PyTorch  |
    | :-----: | :------: | :------: | :------: |
    | 11.7.0  | 8.5.0    |  8.5.2.2 | 2.0.0    |
    | 11.8.0  | 8.6.0    |  8.5.3.1 | 2.0.0    |
    | ...    | ...   |  ... | ...    |
    
    We will be using the first row as our package manifests.
    
- Install `CUDA`. (One can also follow the official instruction:[`CUDA official website`](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)).
    ```shell
    conda install cuda -c nvidia/label/cuda-11.7.0 # install CUDA 11.7.0
    ```

- Install `cuDNN`.
    - Register for the [`NVIDIA developer program`](https://developer.nvidia.com/login)
    - Go to the cuDNN download site:[`cuDNN download archive`](https://developer.nvidia.com/rdp/cudnn-archive)
    - Click `Download cuDNN v8.5.0 (August 8th, 2022), for CUDA 11.x`
    - Download `Local Installer for Windows (Zip)`
    - Unzip `cudnn-windows-x86_64-8.5.0.96_cuda11-archive.zip`
    - Copy all three folders (`bin`,`include`,`lib`) and paste them to the `CUDA` installation directory `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7`. (NOTE `bin`,`include`,`lib` folders are already exist in the CUDA folder.)

- Install `PyTorch`.
    ```shell
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    ```
- Install `TensorRT`.
    Follow the [Nvidia instruction of installation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip).
    - Go to the [TensorRT download site](https://developer.nvidia.com/nvidia-tensorrt-8x-download).
    - Download `TensorRT 8.5 GA for Windows 10 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7 and 11.8 ZIP Package`.
    - Unzip the folder `TensorRT-8.5.2.2` from `TensorRT-8.5.2.2.Windows10.x86_64.cuda-11.8.cudnn8.6.zip`.
    - Add the `<your install path>\TensorRT-8.5.2.2\lib` into PATH environment variable.
    - Go to the folder `<your install path>\TensorRT-8.5.2.2\python`
    - In command window, input 
        ```shell
        conda activate yolov8 # activate dedicated environment
        pip install tensorrt-8.5.2.2-cp310-none-win_amd64.whl # install tensorrt package to python
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


### Using TensorRT to accelerate (Optional)

If you can't install tensorrt in this way, you can look up this [Nvidia guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-install-guide/index.html#installing-zip)


I have provided '.trt' models, but there's a high probability that you have to transform the '.pt' model to '.trt' model by yourself, because the Tensorrt engines are environment specific. This repo may helpful: [TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)


## 2. How to run the program
just run the `main.py` file with the following command

`python main.py`

After a few seconds, the program will start to run. You can see `Main Start` in the console.

Once you hold the right mouse button or the left mouse button (no matter you hold to aim or start shooting), the program will start to aim at the enemy.

## 3. How to change the settings
You can change the settings in the `args.py` file.

### Some important settings!!!:
- model
    - The default model is for Apex. However, you can train your own model using `train.py`, and switch the model using this setting.
    - There're several model in the "model" dir, you can choose one of them.
        - The `.trt` models are for tensorRT, which is about 4 times faster than the `.pt` models, but with the same accuracy. 
        - Model speed: `8n>8s>8m`
        - Model accuracy: `8n<8s<8m`
- crop_size
    - This setting determines the portion of the screen to be detected. Too high may cause difficulty in detecting little objects.
## Note
This program is only for educational purposes. I am not responsible for any damage caused by this program.

## Reference
[Train image dataset](https://universe.roboflow.com/apex-esoic/apexyolov6)

[TensorRt code](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)
