# Running Transfer Learning Pipeline

The goal of this vision-based workflow is to do transfer learning on images to accomplish different classification tasks that range from binary classification to multiclass classification giving best performance on Intel Hardware utilizing the optimizations that could be done.

The pipeline showcases how transfer learning enabled with Intel optimized TensorFlow could be used for image classification on three domains: sports , medical imaging and remote sensing .The workflow showcases AMX  BF16 in SPR which speeds up the training time significantly, without loss in accuracy.

The workflow uses pretrained SOTA models ( RESNET V1.5) from TF hub and transfers the knowledge from a pretrained domain to a different custom domain achieving required accuracy .

![image](https://github.com/intel-innersource/frameworks.ai.end2end-ai-pipelines.e2e-vision-transfer-learning/assets/99835661/de8d7e76-50e4-42d0-8f83-72fdd96a0888)

![image](https://github.com/intel-innersource/frameworks.ai.end2end-ai-pipelines.e2e-vision-transfer-learning/assets/99835661/bbe35b14-5f75-4d92-bcd1-bcbc360f0443)




## Installation in a new Virtual Environment

### Install conda and create new environment

#### Download Miniconda and install

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

Install conda following the steps.

#### Create environment:

```
conda create -n transfer_learning python=3.8 --yes
conda activate transfer_learning
```

### Install TCMalloc

```
conda install -c conda-forge gperftools -y

Set conda path and LD_PRELOAD path

eg :

CONDA_PREFIX=/home/sdp/miniconda3/envs/inc/
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so"

```

### Download the repo

```
git clone https://github.com/intel/vision-based-transfer-learning-and-inference.git
cd vision-based-transfer-learning-and-inference
git checkout main
```

### Install Required packages

```
pip install -r requirements.txt
```
### Support Matrix for Bare-metal
### Operating Systems

| Name | Version | 
| ------ | ------ |
| RHEL | 8.2 or higher |
| CentOS | 8.2 or higher |
| Ubuntu | 18.04<br>20.04 |

### Processor

| Name | Version | 
| ------ | ------ |
| x86 | x86-64 |

### Software Dependencies

| Name | Version | 
| ------ | ------ |
| numactl | N/A |
| scikit-learn | 1.1.2 |
| tensorflow-datasets | 4.6.0 |
| tensorflow-hub | 0.12.0|
| tensorflow | 2.9.0 |
| numpy | 1.23.2 |
| matplotlib | 3.5.2 |
|tensorflow | 2.10.0|

#### Datasets
  
  ##### Medical Imaging Dataset : Extraction
  
        Dataset is downloaded from tensorflow website when the code is run for the first time.
        The dataset used for this domain is colorectal_histology. More details are at the location : https://www.tensorflow.org/datasets/catalog/colorectal_histology
  
  ##### Remote Sensing Dataset
  
        The dataset used for remote sensing domain is resisc45 from tensorflow datasets. More details are at the location : https://www.tensorflow.org/datasets/catalog/resisc45 
        Download the rar file from https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&cid=5C5E061130630A68&id=5C5E061130630A68%21107&parId=5C5E061130630A68%21112&action=locate 
        Unzip the folder
        Run python script resisc_dataset.py with input directory and output directory where you want the split folders to be present ( train and val)
        Eg:
         python resisc_dataset.py --INDIR=/home/sdp/fahim/txlearn/datasets/NWPU-RESISC45/ --OUTDIR=datasets/resisc45
 
### Usage
```
colorectal_tflearn.py : Runs training and inference for medical imaging pipeline on a single node 
resisc_tflearn.py: Runs training and inference for remote sensing pipeline on a single node
colorectal.sh : Bash script for running training or inference for mdeical imaging pipeline on a single node
resisc.sh: Bash script for running training or inference for remote sensing pipeline on a single node 
```

#### Command Line Arguments

```
--PRECISION - whether to use Mixed_Precision or FP32 precision Options : [FP32(default),Mixed Precision]"
              For Mixed Precion , BF16 is used if supported by hardware , if FP16 is supported it is chosen, if none is supported falls back to FP32
--PLATFORM - To optimize for SPR : [None(default),SPR]"
--inference - whether to run only inference"
--cp  - Specify checkpoint directory for inference"
--OUTPUT_DIR  - Specify output Directory where training checkpoints. graphs need to be saved"
--DATASET_DIR  - Specify dataset Directory; if using custom dataset please have train,val,test folders in dataset directory. 
                 If test dataset is not present validation dataset is used"
 --BATCH_SIZE - Batch Size for training[32(default)]"
 --NUM_EPOCHS  - Num epochs for training[100(default)]"

These options can also be set via export variable

ex : export OUTPUT_DIR="logs/fit/trail" 

````
 

#### To run in SPR 


   ##### 1) Remote Sensing Dataset Training
        a) FP32 : bash resisc.sh --PRECISION FP32 --OUTPUT_DIR "logs/fit/resiscFP32/" --DATASET_DIR datasets/resisc45 --PLATFORM SPR --BATCH_SIZE 256
        b) BF16: bash resisc.sh --PRECISION Mixed_Precision  --OUTPUT_DIR "logs/fit/resiscBF16/" --DATASET_DIR  datasets/resisc45 --PLATFORM SPR --BATCH_SIZE 256
   
   ##### 2) Remote Sensing Dataset Inference
        a) Inference FP32: bash resisc.sh --inference -cp "logs/fit/resiscFP32" --PLATFORM SPR --DATASET_DIR datasets/resisc45
        b) Inference BF16: bash resisc.sh --PRECISION Mixed_Precision --inference -cp "logs/fit/resiscBF16" --PLATFORM SPR --DATASET_DIR datasets/resisc45


   ##### 3) Medical Imaging Dataset Training
        a) FP32 : bash colorectal.sh --PRECISION FP32 --OUTPUT_DIR "logs/fit/colorectalFP32/" --DATASET_DIR datasets/colorectal --PLATFORM SPR
        b) BF16: bash colorectal.sh --PRECISION Mixed_Precision --OUTPUT_DIR "logs/fit/colorectalBF16/" --DATASET_DIR datasets/colorectal --PLATFORM SPR
   
   ##### 4) Medical Imaging Dataset Inference
        a) Inference FP32: bash colorectal.sh --inference -cp "logs/fit/colorectalFP32" --PRECISION FP32 --OUTPUT_DIR "logs/fit/colorectalFP32/" --DATASET_DIR datasets/colorectal --PLATFORM SPR
        b) Inference BF16: bash colorectal.sh --inference -cp "logs/fit/colorectalBF16" --PRECISION Mixed_Precision --OUTPUT_DIR "logs/fit/colorectalBF16/" --DATASET_DIR datasets/colorectal --PLATFORM SPR

