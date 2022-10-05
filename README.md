# Semantic Segmentation under Adverse Conditions: A Weather and Nighttime-aware Synthetic Data-based Approach

![alt text](https://github.com/A-Kerim/Semantic-Segmentation-under-Adverse-Conditions/blob/5a8c907a614aa1f72a717ce92b019cdefd835f02/assets/teaser.png "Teaser Figure")

This repository contains the original implementation of the paper Semantic [Segmentation under Adverse Conditions: A Weather and Nighttime-aware Synthetic Data-based Approach](https://bmvc2022.org/programme/papers/), published at the BMVC 2022.

Recent semantic segmentation models perform well under standard weather conditions and sufficient illumination but struggle with adverse weather conditions and nighttime. Collecting and annotating training data under these conditions is expensive, time-consuming, error-prone, and not always practical. Usually, synthetic data is used as a feasible data source to increase the amount of training data. However, just directly using synthetic data may actually harm the model’s performance under normal weather conditions while getting only small gains in adverse situations. Therefore, we present a novel architecture specifically designed for using synthetic training data. We propose a simple yet powerful addition to DeepLabV3+ by using weather and time-of-the-day su-
pervisors trained with multi-task learning, making it both weather and nighttime aware, which improves its mIoU accuracy by 14 percentage points on the ACDC dataset while maintaining a score of 75% mIoU on the Cityscapes dataset.

---

:wrench: Environment Setup
===
To reproduce our experiments, please follow these steps:

### 1. Make sure you have the requirements

  - [Python](https://www.python.org/) (>=3.8)
  - [PyTorch](https://pytorch.org/) (=1.10.1) # Maybe it works with other versions too!

### 2. Clone the repo and install the dependencies
   ```bash
   git clone https://github.com/verlab/TextDrivenVideoAcceleration_TPAMI_2022.git
   cd TextDrivenVideoAcceleration_TPAMI_2022
   pip install -r requirements.txt
   ```

I will explain how to setup and use our code.
---



:movie_camera: Datasets and Simulator
===
* The AWSS dataset can be downloaded using this link: [AWSS](https://www.kaggle.com/datasets/abdulrahmankerim/semantic-segmentation-under-adverse-conditions).
* The Cityscapes and ACDC datasets can be downloaded from these links: [Cityscapes](https://www.cityscapes-dataset.com/) and [ACDC](https://acdc.vision.ee.ethz.ch/).
* The Silver simulator can be downloaded using this link: [Silver](https://livelancsac-my.sharepoint.com/:u:/g/personal/kerim_lancaster_ac_uk/EZFZP1An4B9PmHKDEhaxjGYBWfVXfD8Kfu-yvPOaBpXg8w?e=f0MECt).
---

:e-mail: Contact
===
* Abdulrahman Kerim - PhD Candidate - Lancaster University - a.kerim@lancaster.ac.uk
* Felipe Chamone - PhD Candidate - Universidade Federal de Minas Gerais - cadar@dcc.ufmg.br
* Washington Ramos - PhD Candidate - UFMG - washington.ramos@dcc.ufmg.br
* Leandro Soriano Marcolino - Lecturer at Lancaster University - l.marcolino@lancaster.ac.uk
* Erickson R. do Nascimento - Principal Investigator - UFMG - erickson@dcc.ufmg.br
* Richard Jiang - Associate Professor at Lancaster University - r.jiang2@lancaster.ac.uk> 
---
:memo: Citing 
===
If you find this code useful for your research, please cite the paper: 
```
@inproceedings{kerim2022Semantic,
  title={Semantic Segmentation under Adverse Conditions: A Weather and Nighttime-aware Synthetic Data-based Approach},
  author={Kerim, Abdulrahman and Chamone, Felipe and Ramos, Washington LS and Marcolino, Leandro Soriano and Nascimento, Erickson R and Jiang, Richard},
  booktitle={33nd British Machine Vision Conference 2022, BMVC 2022},
  year={2022}
}
```
----
Acknowledgements
===
This work was funded by the Faculty of Science and Technology of Lancaster University. We thank the High End Computing facility of Lancaster University for the computing resources. The authors would also like to thank CAPES and CNPq for funding different parts of this work.

:shield: License
===
Project is distributed under MIT License
