# HCDG: A Hierarchical Consistency Framework for Domain Generalization on Medical Image Segmentation

## Requirements
-   python 3.6.8
   
   ``` bash
   conda create -n HCDG python=3.6.8 
   ```
   
-   PyTorch 1.5.0 
    
    ``` bash
    conda activate HCDG 
    conda install pytorch==1.5.0 torchvision cudatoolkit=9.2 -c pytorch 
    pip install tensorboardX==2.0
    pip install opencv-python
    pip install pyyaml
    pip install MedPy
    conda install -c anaconda scikit-image
    ```
    

## Usage
1. Download the [Fundus dataset](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view?usp=sharing) into your own folder and change `--data-dir` correspondingly.

2. Train the model.

    ``` bash
    python train.py -g 0 --datasetTrain 1 2 3 --datasetTest 4 --batch-size 4 --resume ./pretrained-weight/test4-40.pth.tar # You need to pretrain a vanilla model
    ```
3. Test the model.

    ``` bash
    python test.py --model-file ./logs/test4/20210910_215812.079473/checkpoint_50.pth.tar --datasetTest 4 -g 0

    ```


