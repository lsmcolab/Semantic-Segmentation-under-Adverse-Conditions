# Semantic Segmentation under Adverse Conditions: A Weather and Nighttime-aware Synthetic Data-based Approach

![alt text](https://github.com/A-Kerim/Semantic-Segmentation-under-Adverse-Conditions/blob/5a8c907a614aa1f72a717ce92b019cdefd835f02/assets/teaser.png "Teaser Figure")

This repository contains the original implementation of the paper Semantic Segmentation under Adverse Conditions: A Weather and Nighttime-aware Synthetic Data-based Approach, published at the BMVC 2022.

Recent semantic segmentation models perform well under standard weather conditions and sufficient illumination but struggle with adverse weather conditions and nighttime. Collecting and annotating training data under these conditions is expensive, time-consuming, error-prone, and not always practical. Usually, synthetic data is used as a feasible data source to increase the amount of training data. However, just directly using synthetic data may actually harm the model’s performance under normal weather conditions while getting only small gains in adverse situations. Therefore, we present a novel architecture specifically designed for using synthetic training data. We propose a simple yet powerful addition to DeepLabV3+ by using weather and time-of-the-day su-
pervisors trained with multi-task learning, making it both weather and nighttime aware, which improves its mIoU accuracy by 14 percentage points on the ACDC dataset while maintaining a score of 75% mIoU on the Cityscapes dataset.

If you find this code useful for your research, please cite the paper: 
```
@inproceedings{kerim2022Semantic,
  title={Semantic Segmentation under Adverse Conditions: A Weather and Nighttime-aware Synthetic Data-based Approach},
  author={Kerim, Abdulrahman and Chamone, Felipe and Ramos, Washington LS and Marcolino, Leandro Soriano and Nascimento, Erickson R and Jiang, Richard},
  booktitle={33nd British Machine Vision Conference 2022, BMVC 2022},
  year={2022}
}
```
---

Usage :computer:
===


I will explain how to setup and use our code.
---


Training :hourglass_flowing_sand:
===
I will explain how to do the training.
---


Testing :eyes:
===
I will explain how to do the texting.
---

Results :100:
===
I will show some results.
---


Code, Datasets, and Simulator
===
* Our semantic segmentation code can be downloaded using this link: [Click]().
* The AWSS dataset can be downloaded using this link: [Click](https://www.kaggle.com/datasets/abdulrahmankerim/semantic-segmentation-under-adverse-conditions).
* The Silver simulator can be downloaded using this link: [Click]().
---

Contact :e-mail:
===
* Abdulrahman Kerim - PhD Candidate - Lancaster University - a.kerim@lancaster.ac.uk
* Felipe Chamone - PhD Candidate - Universidade Federal de Minas Gerais - cadar@dcc.ufmg.br
* Washington Ramos - PhD Candidate - UFMG - washington.ramos@dcc.ufmg.br
* Leandro Soriano Marcolino - Lecturer at Lancaster University - l.marcolino@lancaster.ac.uk
* Erickson R. do Nascimento - Principal Investigator - UFMG - erickson@dcc.ufmg.br
* Richard Jiang - Associate Professor at Lancaster University - r.jiang2@lancaster.ac.uk> 
---
Acknowledgements
===
Abdulrahman Kerim was supported by the Faculty of Science and Technology - Lancaster University.
