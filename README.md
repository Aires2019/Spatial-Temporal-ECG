# Spatial-Temporal-ECG
The code for paper "STAR: A Spatio-Temporal Dynamic Graph Learning Framework for Automated ECG Classification"
# Requirements
In order to run this project, you need to install the following packages:
* pytorch 1.12
* python 3.8.10
* numpy 1.24.4

# Abstract
The 12-lead electrocardiogram (ECG) is a crucial tool for diagnosing cardiovascular diseases. Traditional ECG-based automated diagnostic methods typically capture spatial and temporal dependencies separately. However, due to the complex spatial-temporal relationships in 12-lead ECGs, they often overlook fine-grained correlations across different leads at various timestamps, which could be enhanced using external knowledge. This limitation hinders the representation of patient-specific information. In this work, we address these challenges using a Spatial-Temporal Dynamic Graph (STAR). STAR models each patient’s ECG as a dynamic Spatial-Temporal Relationship (STR) graph, effectively capturing spatial-temporal dependencies and patient-specific features by integrating ECG data with latent spatial-temporal knowledge. Our method incorporates a patch-wise multi-scale temporal feature extraction module to enhance fine-grained temporal representation. To further capture the comprehensive spatial-temporal dependencies in the STR graph, a dynamic graph learning module with a lead-specific pooling method is proposed. Experiments demonstrate that our approach outperforms state-of-the-art methods across multiple tasks on several multi-label datasets. The code is available at [Alt]([URL](https://github.com/Aires2019/Spatial-Temporal-ECG) "code").

