# KG-ORE Knowledge-guided Open Relation Extraction
## 说明 Introduction
本项目为论文 Knowledge-guided Open Relation Extraction 的源代码，目前该论文还在投稿过程中  
This repo contains the source code of the paper "Knowledge-guided Open Relation Extraction". The paper is still in the process of submission

## 模型概述 Model
模型的结构如下图所示：  
The model's architecture is shown in the figure below:
![image](https://user-images.githubusercontent.com/46928336/120064322-13885680-c09e-11eb-9e8d-1eff28d7b9a2.png)

它使用了Attention机制实现了实体和背景知识与原文信息的融合  
It combines the entity and background knowledge with the origin sentence

## 数据集 COER-linked
将[COER](https://dl.acm.org/doi/10.1145/3162077)数据集链接到wikidata获得COER-linked  
We link COER to wikidata and obtain the COER-linked  
数据集可以通过以下链接下载：  
The dataset can be obtained from this link:

链接：https://pan.baidu.com/s/1giJEutSSRPmwfyA1mog1VQ  
提取码：8kt3

## 实验效果
在COER-linked上，本模型的效果为  
The model's performance on COER-linked is  
Precision:90.41 Recall:89.99 F1:90.20
