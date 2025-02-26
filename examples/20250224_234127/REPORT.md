## Scientific Report: Signal Processing Flaws of Convolutional Neural Networks

**Introduction:**

Convolutional Neural Networks (CNNs) have achieved remarkable success in various fields, particularly in image and signal processing. However, CNNs are not without their limitations. This report provides a comprehensive overview of signal processing flaws inherent in CNNs, as identified and addressed in recent research literature. The report synthesizes findings from diverse papers, covering aspects such as the computational complexity of convolution, difficulties in handling non-Euclidean data, sensitivity to adversarial attacks and noise, spectral biases, and quantization effects.

**1. Computational Complexity and Efficiency**

*   **Convolution as a Bottleneck:** Standard convolution operations are computationally intensive, consuming a significant portion of the computing power in CNNs. This is particularly problematic due to the data movement bottleneck in conventional electronic architectures, leading to energy and memory inefficiencies [[1]].
*   **High Computational Complexity:** Standard CNNs can require significant computational resources, increasing processing time, which is a major obstacle for real-time applications, especially on edge devices [[5]].
*   **Large Memory Footprint:** The size and complexity of standard CNN models can exceed the memory constraints of edge devices [[5]].
*   **High Power Consumption:** Complex CNN models consume more power, creating a challenge for battery-powered edge devices [[5]].

**Proposed Solutions and Mitigation Techniques:**

*   **Photonic Convolution:** Implementing convolution using photonic architectures can provide higher computation density and power efficiency compared to electronic implementations [[1]]. This approach uses a synthetic frequency dimension to perform convolution more efficiently.
*   **Depthwise Separable Convolutions:** Using depthwise separable convolutions can reduce computational complexity and the number of parameters [[5]]. This involves separating spatial and channel-wise convolutions, leading to a lightweight model suitable for resource-constrained environments [[7]].
*   **Quantization:** Reducing model size and computational requirements by using lower-precision numerical representations (e.g., 8-bit integers) [[5]].
*   **Ultra Lite CNN (ULCNN):** Development of CNN architectures tailored for low-resource scenarios [[7]]. The ULCNN architecture uses complex-valued convolution and cross-layer feature fusion to enhance efficiency.
*   **Improving Memory Utilization:** A mapping method allows memory regions to overlap and thus utilize the memory more efficiently [[11]].

**2. Handling Non-Euclidean Data**

*   **Limitations with Graph-Structured Data:** Traditional CNN approaches have limitations when applied to graph-structured data (GCNNs) due to the irregularity and geometric complexity of graphs [[2]].
*   **Inability to Express Meaningful Translation:** Traditional CNNs rely on translation invariance, which is easily defined in grid-like data (images) but difficult to express on graphs.
*   **Limitations of Spectral Filtering:** Many GCNN approaches perform spectral filtering in the Fourier domain of the graph, restricting the representational power of the models.

**Proposed Solutions and Mitigation Techniques:**

*   **Algebraic Signal Processing (ASP):** A generalized framework for convolutional signal processing using Reproducing Kernel Hilbert Spaces (RKHS) and ASP, applicable to data on groups, graphons, and other non-Euclidean domains [[3]].
*   **Topology Adaptive Graph Convolutional Networks (TAGCN):** Addresses some of the flaws by using different-sized filters in each layer, allowing for arbitrary-length polynomials in the graph shift operator [[2]].
*   **DAG Convolutional Network (DCN):** A novel GNN architecture designed specifically for learning from data defined over Directed Acyclic Graphs (DAGs) [[8]]. It addresses issues with nilpotent adjacency matrices and incorporates the inductive bias imposed by the DAG structure.
*   **Algebraic Convolutional Filters:** A novel group algebra convolution that operates directly on the Lie group algebra, eliminating the need for computationally expensive lifting steps [[6]].

**3. Sensitivity to Adversarial Attacks and Noise**

*   **Vulnerability to Adversarial Perturbations:** Standard CNNs are easily fooled by adversarial examples (images with small, carefully crafted perturbations) [[18]]. This is a significant concern for safety-critical applications.
*   **Dependence on High-Frequency Patterns:** Standard CNNs overly rely on high-frequency patterns, making them vulnerable to adversarial attacks [[18]].
*   **Noise Amplification:** Conventional deconvolution algorithms in applications like optical time-domain reflectometry (OTDR) are sensitive to noise and can amplify it [[17]].
*   **Overfitting to Noise:** CNNs can be overfitted to noise and artifacts in the training data, leading to poor generalization [[10]].

**Proposed Solutions and Mitigation Techniques:**

*   **Spatial Frequency CNN (SF-CNN):** Replacing the initial feature extraction layers with a Spatial Frequency (SF) layer using Discrete Cosine Transform (DCT) enhances robustness [[18]].
*   **Adversarial Training:** Training models with adversarial examples to improve robustness against perturbations [[22]].
*   **Regularization Techniques:** Using regularization terms in the loss function to reduce the sensitivity of the network's output to input changes [[22]].
*   **Denoising Autoencoders:** Using a novel CNN architecture, the Novel CNN, designed to mitigate overfitting and improve generalization [[10]].
*   **Total-Variation-Based Denoising (TVD):** Remove the small amplitude high-frequency oscillations that are often observed in original IRCNN models [[14]].

**4. Frequency Domain Biases and Limitations**

*   **Over-Reliance on High-Frequency Components:** Standard CNNs overly rely on high-frequency patterns, making them vulnerable to adversarial attacks [[18]].
*   **Instability in the High-Frequency Range:** CNNs exhibit instabilities associated with high-frequency components [[24]].
*   **Limited Adaptivity:** Lack of adaptivity in original IRCNN models, making them not robust to variations in signal classes [[14]].
*   **Aliasing Artifacts:** Standard CNNs often violate the Sampling Theorem in their down-sampling operations, leading to aliasing artifacts [[21]].
*   **Inability to Directly Handle Group Symmetries:** Standard CNNs struggle when the signal's domain isn't a homogeneous space [[6]].

**Proposed Solutions and Mitigation Techniques:**

*   **Spatial Frequency CNN (SF-CNN):** Replacing the initial feature extraction layers with a Spatial Frequency (SF) layer using Discrete Cosine Transform (DCT) enhances robustness [[18]].
*   **FrequencyLowCut Pooling (FLC):** An aliasing-free down-sampling operation designed to be a "plug-and-play" module for CNNs [[21]].
*   **DCN with Spectral Representation:** The architecture admits a spectral representation, allowing for spectral analysis of the signals and filters [[8]].
*   **Algebraic Convolutional Filters:** A novel group algebra convolution that operates directly on the Lie group algebra, eliminating the need for a computationally costly lifting step [[6]].

**5. Quantization-Related Challenges**

*   **Information Loss:** Quantization, a technique used to reduce model size and computational cost, inevitably leads to information loss [[23]].
*   **Overfitting:** Existing models are also prone to overfitting during the training process, potentially resulting in poor performance [[10]].
*   **Suboptimal Quantization Granularity:** Fixed quantization granularities (layer-wise, channel-wise) don't adapt to the varying distributions of weights within the network, leading to increased quantization error [[25]].
*   **Inference Time Bottlenecks:** Software implementation and hardware can be affected, requiring efficient, low-latency inference [[5]].
*   **Inherent Linearity:** The paper does mention that traditional methods are limited by linearity. If not designed carefully, CNNs can also exhibit certain limitations when processing nonlinear signals [[14]].
*  **Sensitivity to Pruning:** CNNs can be sensitive to pruning, as shown by the impact of the pruning technique on MSE scores [[5]].

**Proposed Solutions and Mitigation Techniques:**

*   **Polarization and Quantization:** Using a nonlinear front end that preprocesses input data before it enters the main CNN aims to attenuate or eliminate adversarial perturbations [[23]].
*   **Multi-Scale Convolution, Attention, and Residue:** Multi-scale convolution to extract heterogeneous features, with attention mechanisms for improved adaptability [[14]].
*   **Adaptive Bit Allocation:** Dynamically allocating more bitrates for significant channels and withdrawing bitrates for negligible channels [[25]].
*   **SHEATH Framework:** Detects noise in intermediate feature maps by comparing the noisy output to a noise-free reference [[25]].
*   **IIRCNN+ Model:** Improved version of Iterative Residual Convolutional Neural Network, incorporating techniques to improve accuracy and stability of signal decomposition [[14]].
*   **Algebraic Convolutional Filters:** Proposed to address limitations in standard CNNs and even some G-CNNs handle signals with Lie group symmetries [[6]].

**6. Data-Related Vulnerabilities**

*   **Data Dependency:** CNN models are trained on specific datasets, and their performance may degrade if applied to data from different environments [[12]].
*   **Dependence on Training Data Patterns:** CNN-based approaches might struggle with signals that deviate significantly from the training distribution [[15]].

**Proposed Solutions and Mitigation Techniques:**

*   **Data Augmentation:** The use of rotation-based data augmentation aims to improve the robustness and generalization ability of the model, which is crucial for scenarios with limited training data [[7]].
*   **Designing Robust Training Datasets:** The paper explicitly addresses that proper training data should be used with the CNNs and that said training datasets have a large impact on the final performance of the CNN [[12]].

**7. Domain-Specific Challenges and Mitigation**

*   **RF-Based HAR Systems (WiFi) Limitations:** Issues with narrow bandwidth leading to limited time resolution, crowded channels, and the loss of signal features during preprocessing [[4]].
*   **EEG Signal Processing:** Overfitting and generalization issues are commonly found and require careful tuning of network architecture and signal normalization [[10]].
*   **Automatic Modulation Classification (AMC):** Standard real-valued CNNs may not be optimally designed for processing complex-valued signals directly, requiring specialized techniques like complex-valued convolutions [[7]].

**Proposed Solutions and Mitigation Techniques:**

*   **HAR-SAnet Architecture:** Uses UWB radio for higher resolution and a lightweight CNN structure with efficient convolutions [[4]].
*   **UWB Radio for Higher Resolution:** A Commercial Off-The-Shelf (COTS) Ultra-Wide Band (UWB) radio module is used, which has a much larger channel bandwidth and higher time resolution than WiFi [[4]].
*   **Signal Processing Module:** A module which consists of phase noise reducing, SNR enhancement and motion detection to improve the quality of signal [[4]].
*   **Complex-Valued (CV) Convolution-Based IQ Channel Fusion (IQCF) Module:** This module aims to efficiently fuse the in-phase and quadrature components of the signal [[7]].
*   **Lightweight CNN Design:** HAR-SAnet uses a lightweight CNN structure with efficient convolutions (depth-wise separable, point-wise grouped, dilated convolutions) to reduce computation and storage complexity [[4]].

**8. Addressing Vulnerabilities in Specific Applications**

*   **Brain-Computer Interfaces (BCIs):** CNNs used in EEG-based BCIs are vulnerable to universal adversarial perturbations (UAPs), small and example-independent alterations to the signal [[20]].
*   **Autonomous Driving Models:** CNN-based regression models are vulnerable to adversarial attacks, where pixel-level perturbations can cause misclassifications [[26]].
*   **Medical Ultrasound Imaging:** Manipulating signal processing steps during ultrasound image reconstruction can fool CNNs designed for fatty liver disease classification [[11]].
*   **Optical Time-Domain Reflectometry (OTDR):** Convolutional architectures in OTDR are affected by sampling clock imperfections and sampling timing offset [[4]].

**Proposed Solutions and Mitigation Techniques:**

*   **Total Loss Minimization (TLM) Approach:** Introduced to generate UAPs and create defenses against adversarial attacks [[20]].
*   **Training directly on radio-frequency (RF) data:** To avoid the signal processing dependency [[11]].
*   **Data Augmentation:** Use random adjustments and phase shifts to enhance models [[5]].

**Conclusion:**

This report has compiled a detailed overview of the signal processing flaws inherent in CNNs. It is essential to acknowledge that while CNNs offer significant advancements in signal processing, they are not without their limitations. Overcoming these signal processing flaws remains an active and crucial area of research, with continuous efforts focused on both theoretical analyses and practical implementations to enhance the performance, robustness, and reliability of CNNs in real-world applications.

### References
[1] Lingling Fan, Kai Wang, Heming Wang, Avik Dutt, Shanhui Fan, *"Experimentally Realizing Convolution Processing in the Photonic  Synthetic Frequency Dimension"*, arXiv preprint:2305.03250v2, 2023.

[2] Matthew Baron, *"Topology and Prediction Focused Research on Graph Convolutional Neural  Networks"*, arXiv preprint:1808.07769v1, 2018.

[3] Alejandro Parada-Mayorga, Leopoldo Agorio, Alejandro Ribeiro, Juan Bazerque, *"Convolutional Filtering with RKHS Algebras"*, arXiv preprint:2411.01341v1, 2024.

[4] Zhe Chen, Chao Cai, Tianyue Zheng, Jun Luo, Jie Xiong, Xin Wang, *"RF-Based Human Activity Recognition Using Signal Adapted Convolutional  Neural Network"*, arXiv preprint:2110.14307v2, 2021.

[5] Mostafa Naseri, Eli De Poorter, Ingrid Moerman, H. Vincent Poor, Adnan Shahid, *"High-Throughput Blind Co-Channel Interference Cancellation for Edge  Devices Using Depthwise Separable Convolutions, Quantization, and Pruning"*, arXiv preprint:2411.12541v1, 2024.

[6] Harshat Kumar, Alejandro Parada-Mayorga, Alejandro Ribeiro, *"Algebraic Convolutional Filters on Lie Group Algebras"*, arXiv preprint:2210.17425v1, 2022.

[7] Lantu Guo, Yu Wang, Yun Lin, Haitao Zhao, Guan Gui, *"Ultra Lite Convolutional Neural Network for Fast Automatic Modulation  Classification in Low-Resource Scenarios"*, arXiv preprint:2208.04659v2, 2022.

[8] Samuel Rey, Hamed Ajorlou, Gonzalo Mateos, *"Convolutional Learning on Directed Acyclic Graphs"*, arXiv preprint:2405.03056v1, 2024.

[9] Arsenia Chorti, David Picard, *"Rate Analysis and Deep Neural Network Detectors for SEFDM FTN Systems"*, arXiv preprint:2103.02306v1, 2021.

[10] Haoming Zhang, Chen Wei, Mingqi Zhao, Haiyan Wu, Quanying Liu, *"A novel convolutional neural network model to remove muscle artifacts  from EEG"*, arXiv preprint:2010.11709v2, 2020.

[11] Petar Jokic, Stephane Emery, Luca Benini, *"Improving Memory Utilization in Convolutional Neural Network  Accelerators"*, arXiv preprint:2007.09963v2, 2020.

[12] Yihao Zang, Xianhao Shen, Shaohua Niu, *"A Method for Detecting Abnormal Data of Network Nodes Based on  Convolutional Neural Network"*, arXiv preprint:2107.07407v1, 2021.

[13] Xicheng Lou, Xinwei Li, Hongying Meng, Jun Hu, Meili Xu, Yue Zhao, Jiazhang Yang, Zhangyong Li, *"EEG-DBNet: A Dual-Branch Network for Temporal-Spectral Decoding in  Motor-Imagery Brain-Computer Interfaces"*, arXiv preprint:2405.16090v3, 2024.

[14] Feng Zhou, Antonio Cicone, Haomin Zhou, *"IRCNN$^{+}$: An Enhanced Iterative Residual Convolutional Neural Network  for Non-stationary Signal Decomposition"*, arXiv preprint:2309.04782v2, 2023.

[15] Feng Zhou, Antonio Cicone, Haomin Zhou, *"RRCNN: A novel signal decomposition approach based on recurrent residue  convolutional neural network"*, arXiv preprint:2307.01725v1, 2023.

[16] Yu-Tung Liu, Kuan-Chen Wang, Rong Chao, Sabato Marco Siniscalchi, Ping-Cheng Yeh, Yu Tsao, *"MSEMG: Surface Electromyography Denoising with a Mamba-based Efficient  Network"*, arXiv preprint:2411.18902v2, 2024.

[17] Hao Wu, Ming Tang, *"Beyond the Limitation of Pulse Width in Optical Time-domain  Reflectometry"*, arXiv preprint:2203.09461v1, 2022.

[18] Keng-Hsin Liao, Chin-Yuan Yeh, Hsi-Wen Chen, Ming-Syan Chen, *"Evaluating Adversarial Robustness in the Spatial Frequency Domain"*, arXiv preprint:2405.06345v1, 2024.

[19] Mojtaba Taherisadr, Mohsen Joneidi, Nazanin Rahnavard, *"EEG Signal Dimensionality Reduction and Classification using Tensor  Decomposition and Deep Convolutional Neural Networks"*, arXiv preprint:1908.10432v1, 2019.

[20] Zihan Liu, Lubin Meng, Xiao Zhang, Weili Fang, Dongrui Wu, *"Universal Adversarial Perturbations for CNN Classifiers in EEG-Based  BCIs"*, arXiv preprint:1912.01171v5, 2019.

[21] Julia Grabinski, Steffen Jung, Janis Keuper, Margret Keuper, *"FrequencyLowCut Pooling -- Plug & Play against Catastrophic Overfitting"*, arXiv preprint:2204.00491v2, 2022.

[22] Linhai Ma, Liang Liang, *"Enhance CNN Robustness Against Noises for Classification of 12-Lead ECG  with Variable Length"*, arXiv preprint:2008.03609v4, 2020.

[23] Can Bakiskan, Soorya Gopalakrishnan, Metehan Cekic, Upamanyu Madhow, Ramtin Pedarsani, *"Polarizing Front Ends for Robust CNNs"*, arXiv preprint:2002.09580v1, 2020.

[24] Michal Byra, Grzegorz Styczynski, Cezary Szmigielski, Piotr Kalinowski, Lukasz Michalowski, Rafal Paluszkiewicz, Bogna Ziarkiewicz-Wroblewska, Krzysztof Zieniewicz, Andrzej Nowicki, *"Adversarial attacks on deep learning models for fatty liver disease  classification by modification of ultrasound image reconstruction method"*, arXiv preprint:2009.03364v1, 2020.

[25] Muneeba Asif, Mohammad Kumail Kazmi, Mohammad Ashiqur Rahman, Syed Rafay Hasan, Soamar Homsi, *"SHEATH: Defending Horizontal Collaboration for Distributed CNNs against  Adversarial Noise"*, arXiv preprint:2409.17279v1, 2024.

[26] Yao Deng, Xi Zheng, Tianyi Zhang, Chen Chen, Guannan Lou, Miryung Kim, *"An Analysis of Adversarial Attacks and Defenses on Autonomous Driving  Models"*, arXiv preprint:2002.02175v1, 2020.

[27] Zirui Xu, Fuxun Yu, Xiang Chen, *"DoPa: A Comprehensive CNN Detection Methodology against Physical  Adversarial Attacks"*, arXiv preprint:1905.08790v4, 2019.

[28] Zirui Xu, Fuxun Yu, Xiang Chen, *"LanCe: A Comprehensive and Lightweight CNN Defense Methodology against  Physical Adversarial Attacks on Embedded Multimedia Applications"*, arXiv preprint:1910.08536v1, 2019.

[29] Yalin E. Sagduyu, Tugba Erpek, *"Adversarial Attacks on LoRa Device Identification and Rogue Signal  Detection with Deep Learning"*, arXiv preprint:2312.16715v1, 2023.

[30] Aakash Kumar, *"Applying adversarial networks to increase the data efficiency and  reliability of Self-Driving Cars"*, arXiv preprint:2202.07815v1, 2022.

[31] Diego Gragnaniello, Francesco Marra, Giovanni Poggi, Luisa Verdoliva, *"Analysis of adversarial attacks against CNN-based image forgery  detectors"*, arXiv preprint:1808.08426v1, 2018.

[32] Liang Chen, Paul Bentley, Kensaku Mori, Kazunari Misawa, Michitaka Fujiwara, Daniel Rueckert, *"Intelligent image synthesis to attack a segmentation CNN using  adversarial learning"*, arXiv preprint:1909.11167v1, 2019.

[33] Tong Steven Sun, Yuyang Gao, Shubham Khaladkar, Sijia Liu, Liang Zhao, Young-Ho Kim, Sungsoo Ray Hong, *"Designing a Direct Feedback Loop between Humans and Convolutional Neural  Networks through Local Explanations"*, arXiv preprint:2307.04036v1, 2023.

[34] Xue Jiang, Xiao Zhang, Dongrui Wu, *"Active Learning for Black-Box Adversarial Attacks in EEG-Based  Brain-Computer Interfaces"*, arXiv preprint:1911.04338v1, 2019.

[35] Byung-Kwan Lee, Junho Kim, Yong Man Ro, *"Mitigating Adversarial Vulnerability through Causal Parameter Estimation  by Adversarial Double Machine Learning"*, arXiv preprint:2307.07250v2, 2023.

[36] Guangsheng Zhang, Bo Liu, Huan Tian, Tianqing Zhu, Ming Ding, Wanlei Zhou, *"How Does a Deep Learning Model Architecture Impact Its Privacy? A  Comprehensive Study of Privacy Attacks on CNNs and Transformers"*, arXiv preprint:2210.11049v3, 2022.

[37] Jindong Gu, *"Explainability and Robustness of Deep Visual Classification Models"*, arXiv preprint:2301.01343v1, 2023.

[38] Shengxi Li, Xinyi Zhao, Ljubisa Stankovic, Danilo Mandic, *"Demystifying CNNs for Images by Matched Filters"*, arXiv preprint:2210.08521v1, 2022.

[39] Luis A. Zavala-Mondragón, Peter H. N. de With, Fons van der Sommen, *"A signal processing interpretation of noise-reduction convolutional  neural networks"*, arXiv preprint:2307.13425v1, 2023.

[40] Xi Zhang, Xiaolin Wu, *"On Numerosity of Deep Neural Networks"*, arXiv preprint:2011.08674v1, 2020.

[41] Andrii Skliar, Maurice Weiler, *"Hyperbolic Convolutional Neural Networks"*, arXiv preprint:2308.15639v1, 2023.

[42] Xue Yang, Changchun Bao, *"Embedding Recurrent Layers with Dual-Path Strategy in a Variant of  Convolutional Network for Speaker-Independent Speech Separation"*, arXiv preprint:2203.13574v2, 2022.

[43] Xin Zhang, Yingze Song, Tingting Song, Degang Yang, Yichen Ye, Jie Zhou, Liming Zhang, *"LDConv: Linear deformable convolution for improving convolutional neural  networks"*, arXiv preprint:2311.11587v3, 2023.

[44] Oskar Sjögren, Gustav Grund Pihlgren, Fredrik Sandin, Marcus Liwicki, *"Identifying and Mitigating Flaws of Deep Perceptual Similarity Metrics"*, arXiv preprint:2207.02512v1, 2022.

[45] Jaskaran Singh Walia, Aryan Odugoudar, *"Vulnerability analysis of captcha using Deep learning"*, arXiv preprint:2302.09389v2, 2023.

[46] Minh Tran, Viet-Khoa Vo-Ho, Kyle Quinn, Hien Nguyen, Khoa Luu, Ngan Le, *"CapsNet for Medical Image Segmentation"*, arXiv preprint:2203.08948v1, 2022.

[47] Pei-Chang Guo, *"Regularization for convolutional kernel tensors to avoid unstable  gradient problem in convolutional neural networks"*, arXiv preprint:2102.04294v1, 2021.

[48] Kai Ma, Pengcheng Xi, Karim Habashy, Ashkan Ebadi, Stéphane Tremblay, Alexander Wong, *"Towards Trustworthy Healthcare AI: Attention-Based Feature Learning for  COVID-19 Screening With Chest Radiography"*, arXiv preprint:2207.09312v1, 2022.

[49] Shailesh Arya, Hrithik Mesariya, Vishal Parekh, *"Smart Attendance System Usign CNN"*, arXiv preprint:2004.14289v1, 2020.

[50] Michael Kohler, Adam Krzyzak, Benjamin Walter, *"Analysis of the rate of convergence of an over-parametrized  convolutional neural network image classifier learned by gradient descent"*, arXiv preprint:2405.07619v1, 2024.

[51] Guoming Li, Jian Yang, Shangsong Liang, Dongsheng Luo, *"Spectral GNN via Two-dimensional (2-D) Graph Convolution"*, arXiv preprint:2404.04559v1, 2024.

[52] Ashim Dahal, Saydul Akbar Murad, Nick Rahimi, *"Efficiency Bottlenecks of Convolutional Kolmogorov-Arnold Networks: A  Comprehensive Scrutiny with ImageNet, AlexNet, LeNet and Tabular  Classification"*, arXiv preprint:2501.15757v2, 2025.

[53] Ricard Durall, Margret Keuper, Janis Keuper, *"Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are  Failing to Reproduce Spectral Distributions"*, arXiv preprint:2003.01826v1, 2020.

[54] Hao Wu, Yulong Liu, Wenbin Gao, Xiangrong Xu, *"Uneven illumination surface defects inspection based on convolutional  neural network"*, arXiv preprint:1905.06683v3, 2019.

[55] Tingxuan Huang, Jiacheng Miao, Shizhuo Deng,  Tong, Dongyue Chen, *"Mask-adaptive Gated Convolution and Bi-directional Progressive Fusion  Network for Depth Completion"*, arXiv preprint:2401.07439v2, 2024.

[56] Alessandro Bria, Claudio Marrocco, Francesco Tortorella, *"Sinc-based convolutional neural networks for EEG-BCI-based motor imagery  classification"*, arXiv preprint:2101.10846v1, 2021.

[57] Zhao Zhang, Zemin Tang, Zheng Zhang, Yang Wang, Jie Qin, Meng Wang, *"Fully-Convolutional Intensive Feature Flow Neural Network for Text  Recognition"*, arXiv preprint:1912.06446v2, 2019.

[58] Jianfei Li, Han Feng, Xiaosheng Zhuang, *"Convolutional Neural Networks for Spherical Signal Processing via  Spherical Haar Tight Framelets"*, arXiv preprint:2201.07890v1, 2022.

[59] Alejandro Parada-Mayorga, Alejandro Ribeiro, *"Algebraic Neural Networks: Stability to Deformations"*, arXiv preprint:2009.01433v5, 2020.

[60] Dongmian Zou, Gilad Lerman, *"Graph Convolutional Neural Networks via Scattering"*, arXiv preprint:1804.00099v2, 2018.

[61] Stéphane Mallat, Sixin Zhang, Gaspar Rochette, *"Phase Harmonic Correlations and Convolutional Neural Networks"*, arXiv preprint:1810.12136v2, 2018.

[62] Byeong Tak Lee, Yong-Yeon Jo, Joon-Myoung Kwon, *"Optimizing Neural Network Scale for ECG Classification"*, arXiv preprint:2308.12492v1, 2023.

[63] Michael Perlmutter, Jieqian He, Mark Iwen, Matthew Hirn, *"A Hybrid Scattering Transform for Signals with Isolated Singularities"*, arXiv preprint:2110.04910v1, 2021.

[64] Markus Thom, Franz Gritschneder, *"Rapid Exact Signal Scanning with Deep Convolutional Neural Networks"*, arXiv preprint:1508.06904v5, 2015.

[65] Yimin Yang, Wandong Zhang, Jonathan Wu, Will Zhao, Ao Chen, *"Deconvolution-and-convolution Networks"*, arXiv preprint:2103.11887v1, 2021.

[66] Patricia Pauli, Dennis Gramlich, Frank Allgöwer, *"State space representations of the Roesser type for convolutional layers"*, arXiv preprint:2403.11938v2, 2024.

[67] Xinjue Wang, Esa Ollila, Sergiy A. Vorobyov, *"Graph Convolutional Neural Networks Sensitivity under Probabilistic  Error Model"*, arXiv preprint:2203.07831v5, 2022.

[68] Dung Truong, Scott Makeig, Arnaud Delorme, *"Assessing learned features of Deep Learning applied to EEG"*, arXiv preprint:2111.04309v1, 2021.

[69] Tongyang Xu, Izzat Darwazeh, *"Deep Learning for Over-the-Air Non-Orthogonal Signal Classification"*, arXiv preprint:1911.06174v1, 2019.

[70] Shamik Sarkar, Dongning Guo, Danijela Cabric, *"RadYOLOLet: Radar Detection and Parameter Estimation Using YOLO and  WaveLet"*, arXiv preprint:2309.12094v1, 2023.

[71] Li Su, *"Vocal melody extraction using patch-based CNN"*, arXiv preprint:1804.09202v1, 2018.

[72] James A King, Arshdeep Singh, Mark D. Plumbley, *"Compressing audio CNNs with graph centrality based filter pruning"*, arXiv preprint:2305.03391v1, 2023.

[73] Ender Ozturk, Fatih Erden, Ismail Guvenc, *"RF-Based Low-SNR Classification of UAVs Using Convolutional Neural  Networks"*, arXiv preprint:2009.05519v2, 2020.

[74] Taiwo Oyedare, Vijay K. Shah, Daniel J. Jakubisin, Jeffrey H. Reed, *"Keep It Simple: CNN Model Complexity Studies for Interference  Classification Tasks"*, arXiv preprint:2303.03326v1, 2023.

[75] Qian Chen, Xingjian Dong, Guowei Tu, Dong Wang, Baoxuan Zhao, Zhike Peng, *"TFN: An Interpretable Neural Network with Time-Frequency Transform  Embedded for Intelligent Fault Diagnosis"*, arXiv preprint:2209.01992v2, 2022.

[76] Congmin Fan, Xiaojun Yuan, Ying-Jun Angela Zhang, *"CNN-Based Signal Detection for Banded Linear Systems"*, arXiv preprint:1809.03682v1, 2018.

[77] Ziyu Ye, Andrew Gilman, Qihang Peng, Kelly Levick, Pamela Cosman, Larry Milstein, *"Comparison of Neural Network Architectures for Spectrum Sensing"*, arXiv preprint:1907.07321v1, 2019.

[78] Fuad Noman, Chee-Ming Ting, Sh-Hussain Salleh, Hernando Ombao, *"Short-segment heart sound classification using an ensemble of deep  convolutional neural networks"*, arXiv preprint:1810.11573v1, 2018.

[79] Junru Zhang, Lang Feng, Yang He, Yuhan Wu, Yabo Dong, *"Temporal Convolutional Explorer Helps Understand 1D-CNN's Learning  Behavior in Time Series Classification from Frequency Domain"*, arXiv preprint:2310.05467v1, 2023.

[80] Zhongfan Jia, Chenglong Bao, Kaisheng Ma, *"Exploring Frequency Domain Interpretation of Convolutional Neural  Networks"*, arXiv preprint:1911.12044v2, 2019.

[81] Charles Godfrey, Elise Bishoff, Myles Mckay, Davis Brown, Grayson Jorgenson, Henry Kvinge, Eleanor Byler, *"Testing predictions of representation cost theory with CNNs"*, arXiv preprint:2210.01257v3, 2022.

[82] Zifan Yu, Suya You, Fengbo Ren, *"Frequency-domain Learning for Volumetric-based 3D Data Perception"*, arXiv preprint:2302.08595v2, 2023.

[83] Peter Lorenz, Paula Harder, Dominik Strassel, Margret Keuper, Janis Keuper, *"Detecting AutoAttack Perturbations in the Frequency Domain"*, arXiv preprint:2111.08785v3, 2021.

[84] Roshan Reddy Yedla, Shiv Ram Dubey, *"On the Performance of Convolutional Neural Networks under High and Low  Frequency Information"*, arXiv preprint:2011.06496v1, 2020.

[85] Zhendong Zhang, *"Frequency Pooling: Shift-Equivalent and Anti-Aliasing Downsampling"*, arXiv preprint:2109.11839v1, 2021.

[86] Xiaohan Zhu, Zhen Cui, Tong Zhang, Yong Li, Jian Yang, *"Going Deeper in Frequency Convolutional Neural Network: A Theoretical  Perspective"*, arXiv preprint:2108.05690v1, 2021.

[87] Hengyue Pan, Yixin Chen, Zhiliang Tian, Peng Qiao, Linbo Qiao, Dongsheng Li, *"TFDMNet: A Novel Network Structure Combines the Time Domain and  Frequency Domain Features"*, arXiv preprint:2401.15949v1, 2024.

[88] Jinlai Ning, Michael Spratling, *"The Importance of Anti-Aliasing in Tiny Object Detection"*, arXiv preprint:2310.14221v1, 2023.

[89] Antônio H. Ribeiro, Thomas B. Schön, *"How Convolutional Neural Networks Deal with Aliasing"*, arXiv preprint:2102.07757v1, 2021.

[90] Hagay Michaeli, Tomer Michaeli, Daniel Soudry, *"Alias-Free Convnets: Fractional Shift Invariance via Polynomial  Activations"*, arXiv preprint:2303.08085v2, 2023.

[91] Adrián Rodríguez-Muñoz, Antonio Torralba, *"Aliasing is a Driver of Adversarial Attacks"*, arXiv preprint:2212.11760v1, 2022.

[92] Qiufu Li, Linlin Shen, Sheng Guo, Zhihui Lai, *"WaveCNet: Wavelet Integrated CNNs to Suppress Aliasing Effect for  Noise-Robust Image Classification"*, arXiv preprint:2107.13335v1, 2021.

[93] Mariaclaudia Nicolai, Raffaella Fiamma Cabini, Diego Ulisse Pizzagalli, *"Classification and regression of trajectories rendered as images via 2D  Convolutional Neural Networks"*, arXiv preprint:2409.18832v1, 2024.

[94] Richard Zhang, *"Making Convolutional Networks Shift-Invariant Again"*, arXiv preprint:1904.11486v2, 2019.

[95] Reo Yoneyama, Atsushi Miyashita, Ryuichi Yamamoto, Tomoki Toda, *"Wavehax: Aliasing-Free Neural Waveform Synthesis Based on 2D Convolution  and Harmonic Prior for Reliable Complex Spectrogram Estimation"*, arXiv preprint:2411.06807v1, 2024.

[96] Adithya Sineesh, Mahesh Raveendranatha Panicker, *"Exploring Novel Pooling Strategies for Edge Preserved Feature Maps in  Convolutional Neural Networks"*, arXiv preprint:2110.08842v1, 2021.

[97] Shengju Qian, Hao Shao, Yi Zhu, Mu Li, Jiaya Jia, *"Blending Anti-Aliasing into Vision Transformer"*, arXiv preprint:2110.15156v1, 2021.

[98] Xueyan Zou, Fanyi Xiao, Zhiding Yu, Yong Jae Lee, *"Delving Deeper into Anti-aliasing in ConvNets"*, arXiv preprint:2008.09604v1, 2020.

[99] Shashank Agnihotri, Julia Grabinski, Margret Keuper, *"Improving Feature Stability during Upsampling -- Spectral Artifacts and  the Importance of Spatial Context"*, arXiv preprint:2311.17524v2, 2023.

[100] Zhen Qu, Xian Tao, Fei Shen, Zhengtao Zhang, Tao Li, *"Investigating Shift Equivalence of Convolutional Neural Networks in  Industrial Defect Segmentation"*, arXiv preprint:2309.16902v1, 2023.

[101] Samarth Sinha, Animesh Garg, Hugo Larochelle, *"Curriculum By Smoothing"*, arXiv preprint:2003.01367v5, 2020.

[102] Yu-Chien Lin, Yan Xin, Ta-Sung Lee,  Charlie,  Zhang, Zhi Ding, *"Physics-Inspired Deep Learning Anti-Aliasing Framework in Efficient  Channel State Feedback"*, arXiv preprint:2403.08133v1, 2024.

[103] Yuqing Liu, Qi Jia, Jian Zhang, Xin Fan, Shanshe Wang, Siwei Ma, Wen Gao, *"Hierarchical Similarity Learning for Aliasing Suppression Image  Super-Resolution"*, arXiv preprint:2206.03361v1, 2022.

[104] Michael T. McCann, Kyong Hwan Jin, Michael Unser, *"A Review of Convolutional Neural Networks for Inverse Problems in  Imaging"*, arXiv preprint:1710.04011v1, 2017.

[105] Nathaniel Chodosh, Simon Lucey, *"When to Use Convolutional Neural Networks for Inverse Problems"*, arXiv preprint:2003.13820v1, 2020.

[106] Anadi Chaman, Ivan Dokmanić, *"Truly shift-equivariant convolutional neural networks with adaptive  polyphase upsampling"*, arXiv preprint:2105.04040v3, 2021.

[107] Michael Unser, Stanislas Ducotterd, *"Parseval Convolution Operators and Neural Networks"*, arXiv preprint:2408.09981v1, 2024.

[108] Emil Y. Sidky, Iris Lorente, Jovan G. Brankov, Xiaochuan Pan, *"Do CNNs solve the CT inverse problem?"*, arXiv preprint:2005.10755v1, 2020.

[109] Vincenzo Liguori, *"Pyramid Vector Quantization and Bit Level Sparsity in Weights for  Efficient Neural Networks Inference"*, arXiv preprint:1911.10636v1, 2019.

[110] Seongsik Park, Seijoon Kim, Seil Lee, Ho Bae, Sungroh Yoon, *"Quantized Memory-Augmented Neural Networks"*, arXiv preprint:1711.03712v1, 2017.

[111] Shuchang Zhou, Yuzhi Wang, He Wen, Qinyao He, Yuheng Zou, *"Balanced Quantization: An Effective and Efficient Approach to Quantized  Neural Networks"*, arXiv preprint:1706.07145v1, 2017.

[112] Zhihang Yuan, Chenhao Xue, Yiqi Chen, Qiang Wu, Guangyu Sun, *"PTQ4ViT: Post-training quantization for vision transformers with twin  uniform quantization"*, arXiv preprint:2111.12293v3, 2021.

[113] Bram-Ernst Verhoef, Nathan Laubeuf, Stefan Cosemans, Peter Debacker, Ioannis Papistas, Arindam Mallik, Diederik Verkest, *"FQ-Conv: Fully Quantized Convolution for Efficient and Accurate  Inference"*, arXiv preprint:1912.09356v1, 2019.

[114] Qian Lou, Feng Guo, Lantao Liu, Minje Kim, Lei Jiang, *"AutoQ: Automated Kernel-Wise Neural Network Quantization"*, arXiv preprint:1902.05690v3, 2019.

[115] Jie Hu, Mengze Zeng, Enhua Wu, *"Bag of Tricks with Quantized Convolutional Neural Networks for image  classification"*, arXiv preprint:2303.07080v1, 2023.

[116] Yi-Te Hsu, Yu-Chen Lin, Szu-Wei Fu, Yu Tsao, Tei-Wei Kuo, *"A study on speech enhancement using exponent-only floating point  quantized neural network (EOFP-QNN)"*, arXiv preprint:1808.06474v4, 2018.

[117] Lianqiang Li, Chenqian Yan, Yefei Chen, *"Differentiable Search for Finding Optimal Quantization Strategy"*, arXiv preprint:2404.08010v2, 2024.

[118] Moshe Eliasof, Benjamin Bodner, Eran Treister, *"Haar Wavelet Feature Compression for Quantized Graph Convolutional  Networks"*, arXiv preprint:2110.04824v1, 2021.

[119] Kang-Ho Lee, JoonHyun Jeong, Sung-Ho Bae, *"An Inter-Layer Weight Prediction and Quantization for Deep Neural  Networks based on a Smoothly Varying Weight Hypothesis"*, arXiv preprint:1907.06835v2, 2019.

[120] Ying Nie, Kai Han, Haikang Diao, Chuanjian Liu, Enhua Wu, Yunhe Wang, *"Redistribution of Weights and Activations for AdderNet Quantization"*, arXiv preprint:2212.10200v1, 2022.

[121] Stone Yun, Alexander Wong, *"Do All MobileNets Quantize Poorly? Gaining Insights into the Effect of  Quantization on Depthwise Separable Convolutional Networks Through the Eyes  of Multi-scale Distributional Dynamics"*, arXiv preprint:2104.11849v1, 2021.

[122] Wonyong Sung, Sungho Shin, Kyuyeon Hwang, *"Resiliency of Deep Neural Networks under Quantization"*, arXiv preprint:1511.06488v3, 2015.

[123] Stone Yun, Alexander Wong, *"Where Should We Begin? A Low-Level Exploration of Weight Initialization  Impact on Quantized Behaviour of Deep Neural Networks"*, arXiv preprint:2011.14578v1, 2020.

[124] Zhenhua Liu, Yunhe Wang, Kai Han, Siwei Ma, Wen Gao, *"Post-Training Quantization for Vision Transformer"*, arXiv preprint:2106.14156v1, 2021.

[125] Daria Cherniuk, Stanislav Abukhovich, Anh-Huy Phan, Ivan Oseledets, Andrzej Cichocki, Julia Gusak, *"Quantization Aware Factorization for Deep Neural Network Compression"*, arXiv preprint:2308.04595v1, 2023.

[126] Zhihang Yuan, Yiqi Chen, Chenhao Xue, Chenguang Zhang, Qiankun Wang, Guangyu Sun, *"PTQ-SL: Exploring the Sub-layerwise Post-training Quantization"*, arXiv preprint:2110.07809v2, 2021.

[127] Febin Sunny, Mahdi Nikdast, Sudeep Pasricha, *"A Silicon Photonic Accelerator for Convolutional Neural Networks with  Heterogeneous Quantization"*, arXiv preprint:2205.11244v1, 2022.

[128] Yingzhen Yang, Jiahui Yu, Nebojsa Jojic, Jun Huan, Thomas S. Huang, *"FSNet: Compression of Deep Convolutional Neural Networks by Filter  Summary"*, arXiv preprint:1902.03264v3, 2019.

[129] Clemens JS Schaefer, Siddharth Joshi, Shan Li, Raul Blazquez, *"Edge Inference with Fully Differentiable Quantized Mixed Precision  Neural Networks"*, arXiv preprint:2206.07741v2, 2022.

[130] Xinheng Liu, Yao Chen, Prakhar Ganesh, Junhao Pan, Jinjun Xiong, Deming Chen, *"HiKonv: High Throughput Quantized Convolution With Novel Bit-wise  Management and Computation"*, arXiv preprint:2112.13972v1, 2021.

[131] Bohan Zhuang, Jing Liu, Mingkui Tan, Lingqiao Liu, Ian Reid, Chunhua Shen, *"Effective Training of Convolutional Neural Networks with Low-bitwidth  Weights and Activations"*, arXiv preprint:1908.04680v3, 2019.

[132] Chen Wu, Mingyu Wang, Xiayu Li, Jicheng Lu, Kun Wang, Lei He, *"Phoenix: A Low-Precision Floating-Point Quantization Oriented  Architecture for Convolutional Neural Networks"*, arXiv preprint:2003.02628v1, 2020.

[133] Guangli Li, Lei Liu, Xueying Wang, Xiu Ma, Xiaobing Feng, *"LANCE: Efficient Low-Precision Quantized Winograd Convolution for Neural  Networks Based on Graphics Processing Units"*, arXiv preprint:2003.08646v3, 2020.

[134] Jing Liu, Jianfei Cai, Bohan Zhuang, *"Sharpness-aware Quantization for Deep Neural Networks"*, arXiv preprint:2111.12273v5, 2021.

[135] Shahaf E. Finder, Yair Zohav, Maor Ashkenazi, Eran Treister, *"Wavelet Feature Maps Compression for Image-to-Image CNNs"*, arXiv preprint:2205.12268v4, 2022.

[136] Sungho Shin, Kyuyeon Hwang, Wonyong Sung, *"Quantized neural network design under weight capacity constraint"*, arXiv preprint:1611.06342v1, 2016.

[137] Shahriar Rezghi Shirsavar, Mohammad-Reza A. Dehaqani, *"A Faster Approach to Spiking Deep Convolutional Neural Networks"*, arXiv preprint:2210.17442v1, 2022.

[138] Ido Ben-Yair, Gil Ben Shalom, Moshe Eliasof, Eran Treister, *"Quantized Convolutional Neural Networks Through the Lens of Partial  Differential Equations"*, arXiv preprint:2109.00095v2, 2021.

[139] Raghuraman Krishnamoorthi, *"Quantizing deep convolutional networks for efficient inference: A  whitepaper"*, arXiv preprint:1806.08342v1, 2018.

[140] Ghouthi Boukli Hacene, Vincent Gripon, Matthieu Arzel, Nicolas Farrugia, Yoshua Bengio, *"Quantized Guided Pruning for Efficient Hardware Implementations of  Convolutional Neural Networks"*, arXiv preprint:1812.11337v1, 2018.

[141] Sepehr Eghbali, Ladan Tahvildari, *"Deep Spherical Quantization for Image Search"*, arXiv preprint:1906.02865v1, 2019.

[142] Zhisheng Zhong, Hiroaki Akutsu, Kiyoharu Aizawa, *"Channel-Level Variable Quantization Network for Deep Image Compression"*, arXiv preprint:2007.12619v1, 2020.

[143] Wesley Cooke, Zihao Mo, Weiming Xiang, *"Guaranteed Quantization Error Computation for Neural Network Model  Compression"*, arXiv preprint:2304.13812v1, 2023.

[144] Sungho Shin, Yoonho Boo, Wonyong Sung, *"Fixed-point optimization of deep neural networks with adaptive step size  retraining"*, arXiv preprint:1702.08171v1, 2017.

[145] Yuhui Xu, Yongzhuang Wang, Aojun Zhou, Weiyao Lin, Hongkai Xiong, *"Deep Neural Network Compression with Single and Multiple Level  Quantization"*, arXiv preprint:1803.03289v2, 2018.

[146] Jaehyeon Moon, Dohyung Kim, Junyong Cheon, Bumsub Ham, *"Instance-Aware Group Quantization for Vision Transformers"*, arXiv preprint:2404.00928v1, 2024.

[147] Zhaofan Qiu, Ting Yao, Tao Mei, *"Deep Quantization: Encoding Convolutional Activations with Deep  Generative Model"*, arXiv preprint:1611.09502v1, 2016.

[148] Bohan Zhuang, Chunhua Shen, Mingkui Tan, Peng Chen, Lingqiao Liu, Ian Reid, *"Structured Binary Neural Networks for Image Recognition"*, arXiv preprint:1909.09934v4, 2019.

[149] Elmira Mousa Rezabeyk, Salar Beigzad, Yasin Hamzavi, Mohsen Bagheritabar, Seyedeh Sogol Mirikhoozani, *"Saliency Assisted Quantization for Neural Networks"*, arXiv preprint:2411.05858v1, 2024.

[150] Mucong Ding, Kezhi Kong, Jingling Li, Chen Zhu, John P Dickerson, Furong Huang, Tom Goldstein, *"VQ-GNN: A Universal Framework to Scale up Graph Neural Networks using  Vector Quantization"*, arXiv preprint:2110.14363v1, 2021.

[151] Eric Lybrand, Rayan Saab, *"A Greedy Algorithm for Quantizing Neural Networks"*, arXiv preprint:2010.15979v2, 2020.

[152] Cheeun Hong, Sungyong Baik, Heewon Kim, Seungjun Nah, Kyoung Mu Lee, *"CADyQ: Content-Aware Dynamic Quantization for Image Super-Resolution"*, arXiv preprint:2207.10345v3, 2022.

[153] Taehoon Kim, YoungJoon Yoo, Jihoon Yang, *"FrostNet: Towards Quantization-Aware Network Architecture Search"*, arXiv preprint:2006.09679v4, 2020.

[154] Junfeng Gong, Cheng Liu, Long Cheng, Huawei Li, Xiaowei Li, *"MCU-MixQ: A HW/SW Co-optimized Mixed-precision Neural Network Design  Framework for MCUs"*, arXiv preprint:2407.18267v1, 2024.

[155] Doyun Kim, Han Young Yim, Sanghyuck Ha, Changgwun Lee, Inyup Kang, *"Convolutional Neural Network Quantization using Generalized Gamma  Distribution"*, arXiv preprint:1810.13329v1, 2018.

[156] Lingchuan Meng, John Brothers, *"Efficient Winograd Convolution via Integer Arithmetic"*, arXiv preprint:1901.01965v1, 2019.

[157] Marios Fournarakis, Markus Nagel, *"In-Hindsight Quantization Range Estimation for Quantized Training"*, arXiv preprint:2105.04246v1, 2021.

[158] Marcelo Gennari, Roger Fawcett, Victor Adrian Prisacariu, *"DSConv: Efficient Convolution Operator"*, arXiv preprint:1901.01928v2, 2019.

[159] Jesper Sören Dramsch, Mikael Lüthje, Anders Nymark Christensen, *"Complex-valued neural networks for machine learning on non-stationary  physical data"*, arXiv preprint:1905.12321v2, 2019.

[160] Mehmet Parlak, *"Use Cases for Time-Frequency Image Representations and Deep Learning  Techniques for Improved Signal Classification"*, arXiv preprint:2302.11093v1, 2023.

[161] Jens Heitkaemper, Joerg Schmalenstroeer, Reinhold Haeb-Umbach, *"Statistical and Neural Network Based Speech Activity Detection in  Non-Stationary Acoustic Environments"*, arXiv preprint:2005.09913v2, 2020.

[162] Mohammad Al-Sa'd, Tuomas Jalonen, Serkan Kiranyaz, Moncef Gabbouj, *"Quadratic Time-Frequency Analysis of Vibration Signals for Diagnosing  Bearing Faults"*, arXiv preprint:2401.01172v2, 2024.

[163] Damian Owerko, Charilaos I. Kanatsoulis, Jennifer Bondarchuk, Donald J. Bucci Jr, Alejandro Ribeiro, *"Transferability of Convolutional Neural Networks in Stationary Learning  Tasks"*, arXiv preprint:2307.11588v1, 2023.

[164] Muhammed Saleem, Alec Gunny, Chia-Jui Chou, Li-Cheng Yang, Shu-Wei Yeh, Andy H. Y. Chen, Ryan Magee, William Benoit, Tri Nguyen, Pinchen Fan, Deep Chatterjee, Ethan Marx, Eric Moreno, Rafia Omer, Ryan Raikman, Dylan Rankin, Ritwik Sharma, Michael Coughlin, Philip Harris, Erik Katsavounidis, *"Demonstration of Machine Learning-assisted real-time noise regression in  gravitational wave detectors"*, arXiv preprint:2306.11366v1, 2023.

[165] Takahiro S. Yamamoto, Andrew L. Miller, Magdalena Sieniawska, Takahiro Tanaka, *"Assessing the impact of non-Gaussian noise on convolutional neural  networks that search for continuous gravitational waves"*, arXiv preprint:2206.00882v2, 2022.

[166] Tianxiang Zhan, Yuanpeng He, Yong Deng, Zhen Li, *"Differential Convolutional Fuzzy Time Series Forecasting"*, arXiv preprint:2305.08890v2, 2023.

[167] Alistair McLeod, Damon Beveridge, Linqing Wen, Andreas Wicenec, *"Binary Neutron Star Merger Search Pipeline Powered by Deep Learning"*, arXiv preprint:2409.06266v2, 2024.

[168] Chengyu Zheng, Yuan Zhou, Xiulian Peng, Yuan Zhang, Yan Lu, *"Time-Variance Aware Real-Time Speech Enhancement"*, arXiv preprint:2302.13063v1, 2023.

[169] Georgios Zoumpourlis, Alexandros Doumanoglou, Nicholas Vretos, Petros Daras, *"Non-linear Convolution Filters for CNN-based Learning"*, arXiv preprint:1708.07038v1, 2017.

[170] Felix Juefei-Xu, Vishnu Naresh Boddeti, Marios Savvides, *"Local Binary Convolutional Neural Networks"*, arXiv preprint:1608.06049v2, 2016.

[171] Thomas Wiatowski, Helmut Bölcskei, *"A Mathematical Theory of Deep Convolutional Neural Networks for Feature  Extraction"*, arXiv preprint:1512.06293v3, 2015.

[172] Lenaic Chizat, Edouard Oyallon, Francis Bach, *"On Lazy Training in Differentiable Programming"*, arXiv preprint:1812.07956v5, 2018.

[173] Rickard Brüel Gabrielsson, Gunnar Carlsson, *"Exposition and Interpretation of the Topology of Neural Networks"*, arXiv preprint:1810.03234v3, 2018.

[174] Gavneet Singh Chadha, Andreas Schwung, *"Learning the Non-linearity in Convolutional Neural Networks"*, arXiv preprint:1905.12337v1, 2019.

[175] Junaid Malik, Serkan Kiranyaz, Moncef Gabbouj, *"Operational vs Convolutional Neural Networks for Image Denoising"*, arXiv preprint:2009.00612v1, 2020.

[176] Reinhard Heckel, Paul Hand, *"Deep Decoder: Concise Image Representations from Untrained  Non-convolutional Networks"*, arXiv preprint:1810.03982v2, 2018.

[177] Melikasadat Emami, Mojtaba Sahraee-Ardakan, Parthe Pandit, Sundeep Rangan, Alyson K. Fletcher, *"Implicit Bias of Linear RNNs"*, arXiv preprint:2101.07833v1, 2021.

[178] Wei Wang, Liqiang Zhu, *"Reliable Identification of Redundant Kernels for Convolutional Neural  Network Compression"*, arXiv preprint:1812.03608v1, 2018.

[179] Joshua Bowren, *"A Sparse Coding Interpretation of Neural Networks and Theoretical  Implications"*, arXiv preprint:2108.06622v2, 2021.

[180] Mahalakshmi Sabanayagam, Pascal Esser, Debarghya Ghoshdastidar, *"Analysis of Convolutions, Non-linearity and Depth in Graph Neural  Networks using Neural Tangent Kernel"*, arXiv preprint:2210.09809v4, 2022.

[181] Songtao Liu, Rex Ying, Hanze Dong, Lu Lin, Jinghui Chen, Dinghao Wu, *"How Powerful is Implicit Denoising in Graph Neural Networks"*, arXiv preprint:2209.14514v1, 2022.

[182] Reinhard Heckel, Mahdi Soltanolkotabi, *"Compressive sensing with un-trained neural networks: Gradient descent  finds the smoothest approximation"*, arXiv preprint:2005.03991v1, 2020.

[183] Dongmian Zou, Radu Balan, Maneesh Singh, *"On Lipschitz Bounds of General Convolutional Neural Networks"*, arXiv preprint:1808.01415v1, 2018.

[184] Jingyi Shen, Han-Wei Shen, *"An Information-theoretic Visual Analysis Framework for Convolutional  Neural Networks"*, arXiv preprint:2005.02186v1, 2020.

[185] Zhan Gao, Deniz Gunduz, *"Graph Neural Networks over the Air for Decentralized Tasks in Wireless  Networks"*, arXiv preprint:2302.08447v3, 2023.

