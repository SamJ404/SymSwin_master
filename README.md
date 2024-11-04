# SymSwin:A Transformer-Based Super-Resolution Model Focusing on Multi-Scale Context for Remote-Sensing Images #
***
This is the offical pytorch implementation of SymSwin. [[Paper]]() and [[pretrained model]](https://pan.baidu.com/s/1oVb69eNe2Xe-inGQYkosOA) (extraction code: jbv3) are available. Feel free to send emails to SAMantha404@163.com, discussion is welcome 🙌.  
## Contents 📖 ##  
***
>[Brief Introduction](#section1)  
>[Quick Start](#section2)  
>
>>[Training](#section21)  
>>[Testing](#section22)  
>>
>[Detailed Structure of SymSwin](#section3)  
>[Results](#section4)  
>>[Quantitative Results](#section41)  
>>[Visual Results](#section42)
>>
<a id='section1'></a>
## Brief Introduction ✨ ##
***
* **Motivation:** Objects with significant size variations and textures with varying degrees of coarseness and fineness are present in remote sensing data due to the broad observation, bringing challenges to the task. Targeting at this characteristic of remote sensing images, we propose an innovative super-resolution model based on the Swin Transformer to focus on multi-scale context, named SymSwin.  
* **Innovation:** Our backbone integrates `Symmetric Multi-scale Window (SyMW)` mechanism and `Cross Receptive-field Adaptive Attention (CRAA)` module, which is capable of perceiving features with various sizes. Initially, to extract correct context from multi-scale representations, the method adopts SyMW to pay corresponding attention to features with different measures. Subsequently, the CRAA module is introduced to fuse context obtained with different dependence distances, dealing with the ignoration of links among hybrid-size contents. Furthermore, RS data exhibits poor definition, leading to insufficient visual information for solely spatial feature supervision. The `U-shape Wavelet Transform (UWT) loss` is applied to facilitate the training process from frequency domain.  
* **Keywords:** super-resolution; remote sensing; multi-scale representations; Swin-transformers; wavelet transform loss  
<a id='section2'></a>
## Quick Start ✨ ##  
***
<a id='section21'></a>
### Training 💪 ###  
* Prepare environment:  
```
pip install -r requirement.txt
```  
* Prepare dataset information text file:  
To generate HR images information only:  
```
python .data/generate_meta_info.py --input PATH/TO/DATASET --meta_info PATH/TO/META_INFO.TXT
```  
 To generate Hr-LR pair images infomation:  
```
python .data/generate_meta_info_pairdata.py --input_gt PATH/TO/HR --input_lr PATH/TO/LR --meta_info PATH/TO/META_INFO.TXT
```  
* Modify `./options/SymSwin_train.json` before start training. Pay extra attention to parameters:  
```
task: the folder to save checkpoints/train log/training options  
scale: reconstruction scale  
dataset_info: path to meta_info.txt  
net_type: SymSwin
```  
* Train SymSwin model:  
```
python main_train_psnr.py --opt .options/SymSwin_train.json
```  
<a id='section22'></a>
### Testing 💪 ###  
We provide SymSwin checkpoints pretrained on [[NWPU-RESISC45]](https://pan.baidu.com/s/1oVb69eNe2Xe-inGQYkosOA) and [[DIOR]](https://pan.baidu.com/s/1oVb69eNe2Xe-inGQYkosOA). (extraction code: jbv3)  
* Download the checkpoints to `./checkpoints`. (Other folders you want, 'checkpoints' is recommended.)  
* Prepare dataset HR-LR pair information.  
```
python .data/generate_meta_info_pairdata.py --input_gt PATH/TO/HR --input_lr PATH/TO/LR --meta_info PATH/TO/META_INFO.TXT
```  
* Inference on dataset NWPU-RESISC45/DIOR and get PSNR/SSIM:
```
python eval_RESISC45_SymSwin.py --upscale 4 --metainfo_dir PATH/TO/META_INFO.TXT --pretrained_sr ./checkpoints/SymSwin_resiscx4.pth --save_folder ./prediction --folder_name SymSwin_resisc
python eval_DIOR_SymSwin.py --upscale 4 --metainfo_dir PATH/TO/META_INFO.TXT --pretrained_sr ./checkpoints/SymSwin_diorx4.pth --save_folder ./prediction --folder_name SymSwin_dior
```
Upscale is optional among 2/3/4. The checkpoints of SymSwin and the images sizes of HR-LR pairsneed to be matched with the upscale factor.  
* Get perceptual evaluation:  
```
python cal_perceptual_dior_SymSwin.py --clip_model ./pretrained_model/ViT-B-16.pt --metainfo_dir PATH/TO/META_INFO.TXT --preset ./prediction/SymSwin_resisc
python cal_perceptual_RESISC_SymSwin.py --clip_model ./pretrained_model/ViT-B-16.pt --metainfo_dir PATH/TO/META_INFO.TXT --preset ./prediction/SymSwin_dior
```
Clip model checkpoints is provided offically, or you can download [[here]](https://pan.baidu.com/s/1g8FYx9qAIUbEI1lHo4J-UQ) (extraction code: 2072).  
<a id='section3'></a>
## Detialed Structure of SymSwin ✨ ##
***
* **Overview:** Our SymSwin is composed with three primary parts: shallow feature extraction, deep feature extraction and image reconstruction, as illustrated in *Figure 1a*. The deep feature extraction module in volving *N* cascading SymSwin groups (SymG), each of which includes a Swin Trans former block with Symmetric Multi-scale Window sizes (SyMWB) and a Cross Receptivefield Adaptive Attention (CRAA) module. The SyMWB architecture consists of a cascade of *M* Swin- DCFF attention layers followed by a convolutional layer integrated with a residual connection, as depicted in *Figure1b*. Notably, every Swin-DCFF attention layer incorporates vanilla shifted-window self-attention (SW-SA) and a modified multi-layer perceptron (MLP) with a depth-wise convolution added between two layers, as depicted in *Figure 1c*.  
![Overview structure](https://github.com/SamJ404/SymSwin/blob/main/illstration/Fig1_overall_strructure.jpg)  
*Figure 1*  
* **SyWM:** We employ the symmetric window sizes expanding and shrinking at double scale. We adjust the window sizes on a block-by-block basis, hence, the window sizes in each block are complied with: ![formulate_SyMW](https://github.com/SamJ404/SymSwin/blob/main/illstration/formulate_SyWM.png). The principle is demonstrated in *Figure 2*.  
![SyMW](https://github.com/SamJ404/SymSwin/blob/main/illstration/Fig2_SyMW.png)
*Figure 2*
* **CRAA:** The  CRAA module involves two parts, the Cross-Receptive-field Attention (CRA) to achieve biased fusion of multi-scale representation, and the Adaptive Feed-Forward network (AFF) to emphasis on promising features. The detailed construction of CRAA is demonstrated in *Figure 3*.
![CRAA](https://github.com/SamJ404/SymSwin/blob/main/illstration/Fig3_CRAA.jpg)
*Figure 3*
* **UWT Loss:** We adopt the luminance channel of the image to realize Stationary Wavelet Transform (SWT). The transform process is illustrated in *Figure 4*.  
![SWT](https://github.com/SamJ404/SymSwin/blob/main/illstration/Fig4_SWT.jpg)
*Figure 4*  
We calculate the distance between the reconstruction image and the high-resolution image in each sub-band and sum the distances by weights, which can be formulate as ![formulate_UWT1](https://github.com/SamJ404/SymSwin/blob/main/illstration/formulate_UWT1.png), where we set *μt = [0.05, 0.025, 0.025, 0.02]* following SWT loss. The UWT distance is defined as ![formulate_UWT2](https://github.com/SamJ404/SymSwin/blob/main/illstration/formulate__UWT2.png).  
<a id='section4'></a>
## Results ✨ ##  
We conducted comparative experiments on NWPU-RESISC45 dataset and DIOR dataset comparing our approach with other mainstream SR algorithms based on transformers. We retrained all the models applying exactly the same training configuration for a fair evaluation.  
<a id='section41'></a>
### Quantitative Results 👀 ###  
***
![quantitative_table](https://github.com/SamJ404/SymSwin/blob/main/illstration/quantitative_table.png)  
<a id='section42'></a>
### Visual Results 👀 ###  
***
* **x4 scale**  
![resiscx4_airplane](https://github.com/SamJ404/SymSwin/blob/main/illstration/resiscx4_airplane365.jpg)  
![resiscx4_terrace](https://github.com/SamJ404/SymSwin/blob/main/illstration/resiscx4_terrace101.jpg)  
![diorx4_12228](https://github.com/SamJ404/SymSwin/blob/main/illstration/diorx4_12228.jpg)  
![diorx4_19237](https://github.com/SamJ404/SymSwin/blob/main/illstration/diorx4_19237.jpg)  
* **x3 scale**
![resiscx3_mobile_home_park](https://github.com/SamJ404/SymSwin/blob/main/illstration/resiscx3_mobile_home_park101.jpg)  
![resiscx3_railway_station](https://github.com/SamJ404/SymSwin/blob/main/illstration/resiscx3_railway_station501.jpg)
![diorx3_18337](https://github.com/SamJ404/SymSwin/blob/main/illstration/diorx3_18337.jpg)  
![diorx3_19840](https://github.com/SamJ404/SymSwin/blob/main/illstration/diorx3_19840.jpg)


