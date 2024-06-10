# DF40: Toward Next-Generation Deepfake Detection

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![Release .10](https://img.shields.io/badge/Release-1.0-brightgreen) ![PyTorch](https://img.shields.io/badge/PyTorch-1.11-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7.2-brightgreen)

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="figures/archi.png" style="max-width:60%;">
</div>

Welcome to our work *DF40*, for next-generation deepfake detection. 


<font size=4><b> Table of Contents </b></font>

- [Features](#-features)
- [Quick Start](#-quick-start)
  - [Installation](#1-installation)
  - [Download Data](#2-download-data)
  - [Preprocessing (optional)](#3-preprocessing-optional)
  - [Rearrangement](#4-rearrangement)
  - [Training (optional)](#4-training-optional)
  - [Evaluation](#5-evaluation)
- [Supported Detectors](#-supported-detectors)
- [Results](#-results)
- [Citation](#-citation)
- [Copyright](#%EF%B8%8F-license)

---


**Highlight-1: DF40 Dataset**. The key features of our proposed **DF40 dataset** are as follows:

> ✅ **Forgery Diversity**: *DF40* comprises **40** distinct deepfake techniques (both representive and SOTA methods are included), facilialting the detection of nowadays' SOTA deepfakes and AIGCs. We provide **10** face-swapping methods, **13** face-reenactment methods, **12** entire face synthesis methods, and **5** face editing.
> 
> ✅ **Forgery Realism**: *DF40* includes realistic deepfake data created by highly popular generation software and methods, *e.g.,* HeyGen, MidJourney, DeepFaceLab, to simulate real-world deepfakes. We even include the just-released DiT, SiT, PixArt-$\alpha$, etc.
> 
> ✅ **Forgery Scale**: *DF40* offers **million-level** deepfake data scale for both images and videos.


**Highlight-2: Our Evaluation**. 

> ✅ **Standardized Evaluations**: *DeepfakeBench* introduces standardized evaluation metrics and protocols to enhance the transparency and reproducibility of performance evaluations.
> 
> ✅ **Extensive Analysis and Insights**: *DeepfakeBench* facilitates an extensive analysis from various perspectives, providing new insights to inspire the development of new technologies.


## 💥 DF Dataset
| Type                    | ID-Number | Method        | Download Link | Visual Examples |
|-------------------------|-----------|---------------|---------------|-----------------|
| Face-swapping (FS)      | 1         | FSGAN         |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EUYXBP-FLfpGqoXNvemgCzIBb5mOp4MLzpMaXkVOnB0wzg?e=DtZmzv)               |  [![fsgan-Example](visual_demos/fsgan.gif)](visual_demos/fsgan.gif)               |
|                         | 2         | FaceSwap      |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EfawT-a5jEFGlOq9h6gKHb8BZjCRlUL175-RCkC4xwjYxw?e=SAHQkh)               |  [![faceswap-Example](visual_demos/faceswap.gif)](visual_demos/faceswap.gif)               |
|                         | 3         | SimSwap       |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EcHhmhM4w2FBjDX1DiQpvk0BSk50dgcAT7TiH5-rPmIIDA?e=Kk9pt9)               |  [![simswap-Example](visual_demos/simswap.gif)](visual_demos/simswap.gif)               |
|                         | 4         | InSwapper     |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EWp5dSXVyiFIjqUAO-pqBwABEDMNi1VOlrAXtiCLaMqoqQ?e=7CQaIB)               |  [![inswap-Example](visual_demos/inswap.gif)](visual_demos/inswap.gif)               |
|                         | 5         | BlendFace     |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/Efhz5ni0hRNMss16Ia-cOSkBhboyQFGxE1xGvmFDc61xXw?e=y1gXSF)               |   [![blendface-Example](visual_demos/blendface.gif)](visual_demos/blendface.gif)              |
|                         | 6         | UniFace       |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EQLpC2UcH7pGpg1kBkweBbAB-AGa0ys_B5GqeIQGQ3SVpw?e=64eRr9)               |   [![uniface-Example](visual_demos/uniface.gif)](visual_demos/uniface.gif)              |
|                         | 7         | MobileSwap    |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EUZ2RNJuIANBp_-xL_qNCa4BdS8T1sPTlHy3TwVVzURgug?e=2Ci41t)               |   [![mobileswap-Example](visual_demos/mobileswap.gif)](visual_demos/mobileswap.gif)              |
|                         | 8         | e4s           |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EcPeAiqVLuVIugvovJEev1EBJamTG6GxKNipsKoi-OYKQw?e=NPagz5)               |   [![e4s-Example](visual_demos/e4s.gif)](visual_demos/e4s.gif)              |
|                         | 9         | FaceDancer    |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ERlrQNqPCEpPnKKyLu7KEYMBateR-IRzdAyhy1nV0NhCaQ?e=R7qznh)               |    [![facedancer-Example](visual_demos/facedancer.gif)](visual_demos/facedancer.gif)             |
|                         | 10        | DeepFaceLab   |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ERLDxXqXjjlPgbpQyZ15mIoBsHqSNvkvpuNfRAbmrnPjbg?e=2PTLBS)               |    [![deepfacelab-Example](visual_demos/deepfacelab.gif)](visual_demos/deepfacelab.gif)            |
| Face-reenactment (FR)   | 11        | FOMM          |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EUws9ZP8DfxInXFBDHO-yZ4BTLdxwvByXam3WtMuvJ-Alg?e=orReX0)               |    [![fomm-Example](visual_demos/fomm.gif)](visual_demos/fomm.gif)             |
|                         | 12        | FS_vid2vid    |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EbPmQ8Et1NdBnApAAKGdl-4BiGhBklD6pQQ3KdYSMt6jNA?e=fRPIbM)               |    [![fs_vid2vid-Example](visual_demos/fs_vid2vid.gif)](visual_demos/fs_vid2vid.gif)             |
|                         | 13        | Wav2Lip       |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EWWh6zIJShZOlFPUAjT7ogABUvJzPnryeiXTlR6ID0j6sQ?e=sOLrgx)               |    [![wav2lip-Example](visual_demos/wav2lip.gif)](visual_demos/wav2lip.gif)             |
|                         | 14        | MRAA          |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ETz0X1hB5CBEm0ADJ3dn-F8Bk5foynvQ4jmHnavZejvhdQ?e=cbqZa0)               |    [![mraa-Example](visual_demos/mraa.gif)](visual_demos/mraa.gif)             |
|                         | 15        | OneShot       |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EZJHNvdyjUFBppW-OHQUg_IBxZvThiGNsplpw1B6XKUCsw?e=izixge)               |    [![oneshot-Example](visual_demos/oneshot.gif)](visual_demos/oneshot.gif)             |
|                         | 16        | PIRender      |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ESSUbXL36MBLpJXRsLYFOrIBF8p71mNeHu6j7BddXj0X_A?e=2H7CYC)               |    [![pirender-Example](visual_demos/pirender.gif)](visual_demos/pirender.gif)             |
|                         | 17        | TPSM         |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EdEl6WJr35xPhoa-KwH2sH0Bg5lPR0EmGqAi5ZmHwenzZA?e=uaEK5y)               |    [![tpsm-Example](visual_demos/tpsm.gif)](visual_demos/tpsm.gif)             |
|                         | 18        | LIA           |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EdQp7ty5MOpOjViCBAQxvrEBC_mNS57QvsEsnrXqqOv0mw?e=leBGZm)               |    [![lia-Example](visual_demos/lia.gif)](visual_demos/lia.gif)             |
|                         | 19        | DaGAN         |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EUxxlMvrS4tMjfRDLxGfhUABzhrQ4BgRoQOK30VuR6T49w?e=fnFeXl)               |    [![dagan-Example](visual_demos/dagan.gif)](visual_demos/dagan.gif)             |
|                         | 20        | SadTalker     |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EabgEcMgF8pJkCfxUAamBikB-hbguKrTuuBmBRfA859tHA?e=GnSTyg)               |    [![sadtalker-Example](visual_demos/sadtalker.gif)](visual_demos/sadtalker.gif)             |
|                         | 21        | MCNet         |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ER18O8NJwu9Fo0w6r0Xz48sBrrBLFt6bIVR_iJyc4T4QHg?e=rg0r0h)               |    [![mcnet-Example](visual_demos/mcnet.gif)](visual_demos/mcnet.gif)             |
|                         | 22        | HyperReenact  |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EcbYB4rIAb5Hs9_HtdSITW0B4SRONFu2wjIT4yvWn2JgUA?e=wIKFVz)               |     [![hyperreenact-Example](visual_demos/hyperreenact.gif)](visual_demos/hyperreenact.gif)            |
|                         | 23        | HeyGen        |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EfUFQs-MZRZOq3fpfbPgmasBXsKQAwwMGxjP8E50OdwqCQ?e=0URXYd)               |     [![heygen-Example](visual_demos/heygen.gif)](visual_demos/heygen.gif)            |
| Entire Face Synthesis (EFS) | 24    | VQGAN         |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EaX03B4zRzBFqtKXaQDAR3gBh18HYKv5q6k0SQvTMYz5OQ?e=H4OqG3)               |     [![vqgan-Example](image_visual/vqgan.png)](image_visual/vqgan.png)            |
|                         | 25        | StyleGAN2     |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EfcmYbPWeC1FkMs2mnpSpkQBCMtODYwuyLMoauHo5KvNSQ?e=UZsjPC)               |     [![stylegan2-Example](image_visual/stylegan2.png)](image_visual/stylegan2.png)            |
|                         | 26        | StyleGAN3     |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ERvrO00gex5Eu-ZW_D_hyKEBjhhXa76aqkTTDN8Q34Izpw?e=R5XZ9N)               |     [![stylegan3-Example](image_visual/stylegan3.png)](image_visual/stylegan3.png)            |
|                         | 27        | StyleGAN-XL   |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ESM4dEPKYspLjMxzWWlO6owBSGwpNWo4XYUuaCzAdbWPPQ?e=g4T7PS)               |     [![styleganxl-Example](image_visual/styleganxl.png)](image_visual/styleganxl.png)            |
|                         | 28        | SD-2.1        |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ESY6AbSIx_hHj-2aLduJrdwBxQwe3U5ANeG0sGFk0PnuOQ?e=Vjwm0n)               |                 |
|                         | 29        | DDPM          |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EVfM0nmd0-tGgEZXoxM_TVEB-fGjJ1X8VeI_FqfDZGdzDA?e=u9R4nG)               |                 |
|                         | 30        | RDDM          |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EQLtPwX7-IZKhY20J4h1S4sBc0bpyRn0MdYBcbBTHiupWg?e=cCbfCR)               |                 |
|                         | 31        | PixArt-$\alpha$ |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ERnzCPmKchdEqlwnyK5zvxQB4sVFQo6wE_h2zwXLVgSb5A?e=mARXUu)             |                 |
|                         | 32        | DiT-XL/2      |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/Eb6NICxFafhFq54pRNlM74sB32e9Tq48hvUF53A5tu1I5Q?e=Nefcld)               |                 |
|                         | 33        | SiT-XL/2      |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EW-3ZFwOh9tPtTvBmpJoHJoBZnLoHv4QS6Bq8CYZPZ5P0w?e=aRBzlv)               |                 |
|                         | 34        | MidJounery6   |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/Ea7gykdNkYxIjdihRkOX-OkB-wfrXRvcyHrL8xO_FrT1Iw?e=kXyI3J)               |      [![mj-Example](image_visual/mj.png)](image_visual/mj.png)           |
|                         | 35        | WhichisReal   |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EfE0VjMMmhVGhaB3NOKkIewBFelxCqJZjbKWEcSGmbYVEg?e=k6cCAZ)               |      [![vqgan-Example](image_visual/whichisreal.png)](image_visual/whichisreal.png)           |
| Face Edit (FE)          | 36        | CollabDiff    |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EXYtnmeNlDlHgiMrLoesrZMBfQaDmX-HaN2-o-DVnZBe7Q?e=y5KC3p)               |                 |
|                         | 37        | e4e           |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EerqA8zZXfpCiuZINnh3PksB8KvKdhuNINJ5mDGraYrkTw?e=wZRPnV)               |                 |
|                         | 38        | StarGAN       |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/Ea6LPDfiPvROm2QrD9yDvUUBHUVQLOlZW1UjufLblzBUBw?e=eAzmJ8)               |       [![stargan-Example](image_visual/stargan.png)](image_visual/stargan.png)          |
|                         | 39        | StarGANv2     |               |                 |
|                         | 40        | StyleCLIP     |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EbKlB_5yBthCrngJl74LSG4BiX6Jwf9ciElzJlD-H-_Vzw?e=eEPuuN)               |       [![styleclip-Example](image_visual/styleclip.png)](image_visual/styleclip.png)          |


## 📚 Detectors
<a href="#top">[Back to top]</a>

We perform evaluations using the following 7 detectors:

⭐️  **Detectors** (**7** detectors):
  - [Xception](./training/detectors/xception_detector.py), [RECCE](./training/detectors/recce_detector.py), [RFM](./training/detectors/rfm_detector.py), [CLIP](./training/detectors/clip_detector.py), [SBI](./training/detectors/sbi_detector.py), [SPSL](./training/detectors/spsl_detector.py), [SRM](./training/detectors/srm_detector.py)


|                  | File name                               | Paper                                                                                                                                                                                                                                                                                                                                                         |
|------------------|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                                                                                                                                             |
| CLIP            | [clip_detector.py](./training/detectors/clip_detector.py)           | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) ICML 2021                                                                                                                                                                                   |                                                                                                                                                                                                                                                                            |
| RFM   | [rfm_detector.py](./training/detectors/rfm_detector.py)       | [Representative Forgery Mining for Fake Face Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Representative_Forgery_Mining_for_Fake_Face_Detection_CVPR_2021_paper.pdf) CVPR 2021                                                                                                                                                                                                                                                                   |
| TimeTransformer    | [timetransformer_detector.py](./training/detectors/timetransfromer_detector.py)         | [Is space-time attention all you need for video understanding?](https://proceedings.mlr.press/v139/bertasius21a/bertasius21a-supp.pdf) ICML 2021                                                       |
| VideoMAE    | [videomae_detector.py](./training/detectors/videomae_detectors.py)         | [Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training](https://proceedings.neurips.cc/paper_files/paper/2022/file/416f9cb3276121c42eebb86352a4354a-Paper-Conference.pdf) NIPS 2022                                                       |
| X-CLIP    | [xclip_detector.py](./training/detectors/xclip_detector.py)         | [Expanding Language-Image Pretrained Models for General Video Recognition](https://arxiv.org/pdf/2208.02816) ECCV 2022                                                       |



## ⏳ Quick Start

### 1. Installation
Please run the following script to install the required libraries:

```
sh install.sh
```

### 3. Preprocessing

<a href="#top">[Back to top]</a>

**❗️Note**: If you want to directly utilize the data, including frames, landmarks, masks, and more, that I have provided above, you can skip the pre-processing step. **However, you still need to run the rearrangement script to generate the JSON file** for each dataset for the unified data loading in the training and testing process.

To start preprocessing your dataset, please follow these steps:

1. Download the [shape_predictor_81_face_landmarks.dat](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat) file. Then, copy the downloaded shape_predictor_81_face_landmarks.dat file into the `./preprocessing/dlib_tools folder`. This file is necessary for Dlib's face detection functionality.

2. Open the [`./preprocessing/config.yaml`](./preprocessing/config.yaml) and locate the line `default: DATASET_YOU_SPECIFY`. Replace `DATASET_YOU_SPECIFY` with the name of the dataset you want to preprocess, such as `FaceForensics++`.

7. Specify the `dataset_root_path` in the config.yaml file. Search for the line that mentions dataset_root_path. By default, it looks like this: ``dataset_root_path: ./datasets``.
Replace `./datasets` with the actual path to the folder where your dataset is arranged. 

Once you have completed these steps, you can proceed with running the following line to do the preprocessing:

```
cd preprocessing

python preprocess.py
```


### 4. Rearrangement
To simplify the handling of different datasets, we propose a unified and convenient way to load them. The function eliminates the need to write separate input/output (I/O) code for each dataset, reducing duplication of effort and easing data management.

After the preprocessing above, you will obtain the processed data (*i.e., frames, landmarks, and masks*) for each dataset you specify. Similarly, you need to set the parameters in `./preprocessing/config.yaml` for each dataset. After that, run the following line:
```
cd preprocessing

python rearrange.py
```
After running the above line, you will obtain the JSON files for each dataset in the `./preprocessing/dataset_json` folder. The rearranged structure organizes the data in a hierarchical manner, grouping videos based on their labels and data splits (*i.e.,* train, test, validation). Each video is represented as a dictionary entry containing relevant metadata, including file paths, labels, compression levels (if applicable), *etc*. 


### 5. Training (TODO)

<a href="#top">[Back to top]</a>

To be released.


### 6. Evaluation
If you only want to evaluate the detectors to produce the results of the cross-dataset evaluation, you can use the the [`test.py`](./training/test.py) code for evaluation. Here is an example:

```
python3 training/test.py \
--detector_path ./training/config/detector/xception.yaml \
--test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "DFDCP" \
--weights_path ./training/weights/xception_best.pth
```
**Note that we have provided the pre-trained weights for each detector (you can download them from the [`link`](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1)).** Make sure to put these weights in the `./training/weights` folder.



<!-- 
## 🛡️ License

<a href="#top">[Back to top]</a>


This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Secure Computing Lab of Big Data (SCLBD) at The School of Data Science (SDS) of The Chinese University of Hong Kong, Shenzhen, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on the research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.

If you have any suggestions, comments, or wish to contribute code or propose methods, we warmly welcome your input. Please contact us at wubaoyuan@cuhk.edu.cn or yanzhiyuan1114@gmail.com. We look forward to collaborating with you in pushing the boundaries of deepfake detection. -->
