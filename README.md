# KAN4Graph

Implementation of Kolmogorov-Arnold Network (KAN) for Graphs



<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Usage">Usage</a></li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
    <li><a href="#citation">Citation</a></li>
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

| Dataset         | Type            |   # Nodes   |    # Edges    | # Feature Dimensions | # Classes |
| --------------- | --------------- | :---------: | :-----------: | :------------------: | :-------: |
| Cora            | Attribute Graph |    2,708    |     5,278     |        1,433         |     7     |
| CiteSeer        | Attribute Graph |    3,327    |     4,614     |        3,703         |     6     |
| Amazon-Photo    | Attribute Graph |    7,650    |    119,081    |         745          |     8     |
| ogbn-arxiv      | Attribute Graph |   169,343   |   1,166,243   |         128          |    40     |
| Reddit          | Attribute Graph |   232,965   |  23,213,838   |         602          |    41     |
| ogbn-products   | Attribute Graph |  2,449,029  |  61,859,140   |         100          |    47     |
| ogbn-papers100M | Attribute Graph | 111,059,956 | 1,615,685,872 |         128          |    172    |



## Acknowledgements

Our code are partly based on the following GitHub repository. Thanks for their awesome works. 
- [pykan](https://github.com/KindXiaoming/pykan): the official implement of KAN.
- [fast-kan](https://github.com/ZiyaoLi/fast-kan): the implement of KAN (fast version). 
- [Awesome Deep Graph Clustering](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering): a collection of deep graph clustering (papers, codes, and datasets). 
- [SCGC](https://github.com/yueliu1999/SCGC): the official implement of Simple Contrastive Graph Clustering (SCGC) model.







## Citations

If you find this repository helpful, please cite our paper (coming soon).

<p align="right">(<a href="#top">back to top</a>)</p>

