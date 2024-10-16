# KAN4Graph

Implementation of Kolmogorov-Arnold Network (KAN) for Graphs. Any communications, collaborations, issues, PRs are welcomed. The contributors will be listed [here](https://github.com/yueliu1999/KAN4Graph?tab=readme-ov-file#Contributors). Please contact yueliu19990731@163.com. If you find this repository useful to your research or work, it is really appreciate to star this repository. :heart:



[![stars](https://img.shields.io/github/stars/yueliu1999/KAN4Graph?color=yellow)](https://github.com/yueliu1999/KAN4Graph/stars)
[![forks](https://img.shields.io/github/forks/yueliu1999/KAN4Graph?color=lightblue)](https://github.com/yueliu1999/KAN4Graph/forks)
[![ issues](https://img.shields.io/github/issues-raw/yueliu1999/KAN4Graph?color=%23FF9600)](https://github.com/yueliu1999/KAN4Graph/issues)
[![ visitors](https://visitor-badge.glitch.me/badge?page_id=yueliu1999.KAN4Graph)](https://github.com/yueliu1999/KAN4Graph)





<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Usage">Usage</a></li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
  </ol>
</details>



## Usage



### Requirements

KAN4Graph is implemented with Python3.8.16 and 1 NVIDIA Tesla V100 SXM2 16 GB



Python package information is summarized in **requirements.txt**:

- torch==1.7.1
- tqdm==4.59.0
- numpy==1.19.2
- munkres==1.1.4
- scikit_learn==1.2.0



### Datasets

| Dataset | Type            | # Nodes | # Feature Dimensions | # Edges | # Classes |
| ------- | --------------- | :-----: | :-----: | :------------------: | :-------: |
| BAT     | Attribute Graph |   131   |   81    |         1038         |     4     |
| UAT     | Attribute Graph |  1,190  |   239   |        13,599        |     4     |
| EAT     | Attribute Graph |  399  |   203   |        5,994	        |     4     |

still updating...



### Quick Start

clone this repository and change directory to KAN4Graph

```
git clone https://github.com/yueliu1999/KAN4Graph.git
cd ./KAN4Graph
```



run codes

```
python train.py
```



### Results

| Dataset | Metric | Score |
| ------- | ------ | ----- |
| BAT     | ACC    | 77.86 |
|         | NMI    | 54.48 |
|         | ARI    | 52.33 |
|         | F1     | 77.34 |
| UAT     | ACC    | 57.05 |
|         | NMI    | 25.49 |
|         | ARI    | 24.97 |
|         | F1     | 55.80 |
| EAT     | ACC    | 57.87 |
|         | NMI    | 34.16 |
|         | ARI    | 27.52 |
|         | F1     | 58.09 |


still updating...



## Acknowledgements

Our code are partly based on the following GitHub repository. Thanks for their awesome works. 
- [pykan](https://github.com/KindXiaoming/pykan): the official implement of KAN.
- [fast-kan](https://github.com/ZiyaoLi/fast-kan): the implement of KAN (fast version). 
- [Awesome Deep Graph Clustering](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering): a collection of deep graph clustering (papers, codes, and datasets). 
- [SCGC](https://github.com/yueliu1999/SCGC): the official implement of Simple Contrastive Graph Clustering (SCGC) model.





## Contributors

<a href="https://github.com/yueliu1999" target="_blank"><img src="https://avatars.githubusercontent.com/u/41297969?s=64&v=4" alt="yueliu1999" width="96" height="96"/></a> 


If you find this repository helpful, please cite our paper.

```
@article{liuyue_SCGC,
  title={Simple contrastive graph clustering},
  author={Liu, Yue and Yang, Xihong and Zhou, Sihang and Liu, Xinwang},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}

@misc{KAN4Graph,
  author = {Yue Liu},
  title = {KAN4Graph},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yueliu1999/KAN4Graph}},
}
```


<p align="right">(<a href="#top">back to top</a>)</p>

