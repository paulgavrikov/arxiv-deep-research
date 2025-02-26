## Scientific Report: Building Small, Generalizable Language Models for Vision-Language Tasks (LVLMs)

**Research Question:** How do I make very small LVLMs that generalize well?

**Introduction:**

The development of Large Language Models (LLMs) has revolutionized Natural Language Processing (NLP), demonstrating remarkable capabilities in various tasks. However, their substantial computational demands and memory footprints hinder deployment in resource-constrained environments. This limitation motivates the research into creating smaller, more efficient language models (LVLMs) that can maintain strong generalization performance across diverse tasks and languages. This report synthesizes findings from multiple research papers to address the question of how to build very small LVLMs that generalize well. The strategies explored encompass architectural choices, data curation techniques, training methodologies, and optimization strategies.

**1. Architectural Choices and Parameter Efficiency:**

*   **Attention Head to Layer Ratio:** Several studies suggest an optimal ratio of attention heads to layers in smaller Transformer models. A ratio of around 2 has been identified as beneficial, contrasting with the 1:1 ratio often found in larger base models [8, 13].
*   **Reduced Embedding Size:** The size of the embedding layer is a critical factor influencing the overall model size. Techniques like byte-level tokenization and efficient tokenizers can significantly reduce the vocabulary size and, consequently, the embedding layer's parameter count [9, 8].
*   **Structured Pruning:** Pruning techniques, especially structured pruning (removing entire neurons, channels, or layers), are essential for achieving real-world efficiency gains [17, 3]. Iterative pruning methods, where the model is gradually pruned and fine-tuned, are often more effective than one-shot pruning [10, 3].
*   **Mixture-of-Experts (MoE) Architectures:** MoEs offer a promising avenue for creating efficient LLMs with a larger effective parameter count without a proportional increase in computational cost [7, 18].
*   **Adapter-Based Approach:** Utilizing adapters to add task-specific knowledge to a small language model without significantly increasing the number of parameters [6, 18, 7].
*   **Factorized Embeddings:** Reducing the parameter count of the embedding layer is crucial. This can be achieved by splitting the embedding layer into two smaller matrices [9, 8].
*   **Character-Based Models:** For low-resource languages or noisy user-generated content, character-based models can offer a more robust alternative to subword-based tokenization [5].
*   **Fine-tuning Stacks of Language Models (FSLM):** This approach can perform better than non-adapter models on benchmarks and shows good interpretability [17].

**2. Data Curation and Augmentation:**

*   **Prioritize Data Quality:** The quality of training data is paramount, often outweighing the sheer volume of data [21]. Synthetically generated datasets, if carefully crafted, can be highly effective [6].
*   **Targeted Data Augmentation:** Employ data augmentation techniques to expand the training data, particularly in low-resource scenarios. Techniques such as back-translation, synonym replacement, and noise injection can improve model robustness [2309.12599v1, 1].
*   **Adversarial Training:** Generating adversarial examples can improve robustness against noise and perturbations in the input data [189, 2309.17954v1].
*   **Leverage LLMs for Data Augmentation:** Use larger LLMs to generate diverse and informative rationales or augmentations for existing training data. This can help transfer knowledge to smaller models and improve their generalization [16, 11].
*   **Instruction Tuning:** Frame training prompts as explicit instructions to guide the model to generate appropriate content [8, 9].
*   **Curriculum Learning:** Implement curriculum learning strategies, where the model is gradually exposed to more complex examples during training [26, 13].
*   **Domain-Specific Data:** Focus on in-domain data is effective for specializing models in a specific subject [5].
*   **Cross-Lingual Training:** Incorporate data from related languages to improve cross-lingual transfer and generalization [2, 4].
*   **Mitigate Bias:** Ensure the training data is unbiased and representative of the target population to avoid perpetuating societal biases [15].
*   **Knowledge Expansion:** Freeze the original layers and add new layers to expand the model's knowledge without catastrophic forgetting [8].

**3. Training and Optimization Methodologies:**

*   **Knowledge Distillation (KD):** Knowledge distillation is a powerful technique for transferring knowledge from larger, more complex models (teachers) to smaller, more efficient models (students) [21, 42]. This includes extracting rationales from LLMs using Chain-of-Thought (CoT) prompting and then training smaller models to predict both labels and rationales [21].
*   **Instruction Fine-tuning:** Instruction fine-tuning is essential for directing the model to learn a specific input-output mapping [9, 13].
*   **Self-Refinement:** A second phase where the SLM self-refines its reasoning abilities using a preference optimization strategy [16].
*   **Multilingual Instruction Tuning:** A small amount of multilingual data significantly improves multilingual instruction-following [15].
*   **Energy-Efficient Hardware:** Consider deploying the models on energy-efficient hardware to reduce energy consumption [7].

**4. Quantization Strategies:**

*   **Quantization Aware Training (QAT):** Training with quantization in the loop to allow the model to adapt to lower precision representations [16, 7]. The paper proposes that the benefits of quantized language models are optimized when aggressive regularization is applied [4, 4].
*   **Post-Training Quantization (PTQ):** Quantizing the model after training to avoid the computational cost of QAT [21, 9].
*   **Mixed-Precision Quantization:** Using different precision levels for different parts of the model to balance accuracy and efficiency [2410.12883v1, 18, 14].
*   **Low-Bit Quantization:** Experiments with the lowest possible bit to increase the number of samples [9].
*   **Weight and Activation Quantization:** Addresses the challenges of representing both weight and activation by reducing their respective bitwidths [14].
*   **Power-of-Two Quantization:** The paper suggests exploring power-of-two (PoT) quantization to simplify multiplications and improve performance on certain types of hardware [14].
*   **Atomic Representations:** Employing atomic IDs for key elements helps the model focus on semantic meaning rather than surface form [14].
*   **Hardware Optimization**: Deploying the models on energy-efficient hardware like Esperanto's ET-SoC-1 chip significantly improves the TPS/W (tokens per second per watt), reducing energy consumption [7].

**5. Pruning Techniques:**

*   **Structured Pruning:** Removing entire structures, such as layers, attention heads, or channels, to improve hardware efficiency [17]. It also preserves the robustness of unstructured pruning by not decreasing the kurtosis of weights [7].
*   **Magnitude-Based Pruning:** Removing weights with the smallest magnitudes, a simple yet effective approach [9, 10]. The technique is also more robust to noise [5].
*   **Knowledge Expansion:** Freezing parameters is essential for avoiding catastrophic forgetting in multi-domain learning [8].
*   **Two-Stage Training:** Is effective. First, train on general data, then fine-tune on task-specific data with instruction following [7].

**6. Hyperparameter Tuning and Regularization:**

*   **Hyperparameter Tuning:** Conduct a thorough hyperparameter search to optimize the model's performance for the specific task and dataset [4]. Pay attention to embedding size, hidden layer size, learning rate, and regularization parameters.
*   **Regularization:** Implement strong regularization techniques, especially those used in AWD-LSTMs (DropConnect, variational dropout, word dropout, L1/L2 regularization) [4].
*   **Early Stopping:** Track the validation loss during training to prevent overfitting and determine when to stop training [4].

**Conclusion:**

Creating small, generalizable LVLMs requires a holistic approach that combines efficient architectures, strategic data curation, and targeted training methodologies. The techniques described in this report, drawn from a range of research papers, offer a valuable foundation for researchers and practitioners seeking to develop resource-efficient models that can perform well across diverse tasks and languages. While individual techniques may have limitations, a careful combination of these strategies can help to achieve the desired balance between model size, performance, and generalization ability in the challenging domain of vision-language modeling.

### References
[1] Ioana Buhnila, Aman Sinha, Mathieu Constant, *"Retrieve, Generate, Evaluate: A Case Study for Medical Paraphrases  Generation with Small Language Models"*, arXiv preprint:2407.16565v1, 2024.

[2] Seonjeong Hwang, Yunsu Kim, Gary Geunbae Lee, *"Cross-lingual Transfer for Automatic Question Generation by Learning  Interrogative Structures in Target Languages"*, arXiv preprint:2410.03197v1, 2024.

[3] Jian Gao, Xiao Zhang, Ji Wu, Miao Li, *"Enhancing elusive clues in knowledge learning by contrasting attention  of language models"*, arXiv preprint:2409.17954v1, 2024.

[4] Stuart Mesham, Luc Hayward, Jared Shapiro, Jan Buys, *"Low-Resource Language Modelling of South African Languages"*, arXiv preprint:2104.00772v1, 2021.

[5] Arij Riabi, Benoît Sagot, Djamé Seddah, *"Can Character-based Language Models Improve Downstream Task Performance  in Low-Resource and Noisy Language Scenarios?"*, arXiv preprint:2110.13658v1, 2021.

[6] Rohan Deepak Ajwani, Zining Zhu, Jonathan Rose, Frank Rudzicz, *"Plug and Play with Prompts: A Prompt Tuning Approach for Controlling  Text Generation"*, arXiv preprint:2404.05143v1, 2024.

[7] Aayush Shah, Shankar Jayaratnam, *"Energy Efficient Protein Language Models: Leveraging Small Language  Models with LoRA for Controllable Protein Generation"*, arXiv preprint:2411.05966v1, 2024.

[8] Ankit Maloo, Abhinav Garg, *"Cross-Domain Content Generation with Domain-Specific Small Language  Models"*, arXiv preprint:2409.17171v2, 2024.

[9] Ben Fauber, *"Pretrained Generative Language Models as General Learning Frameworks for  Sequence-Based Tasks"*, arXiv preprint:2402.05616v1, 2024.

[10] Bumjun Kim, Kunha Lee, Juyeon Kim, Sangam Lee, *"Small Language Models are Equation Reasoners"*, arXiv preprint:2409.12393v1, 2024.

[11] Tom Pieper, Mohamad Ballout, Ulf Krumnack, Gunther Heidemann, Kai-Uwe Kühnberger, *"Enhancing SLM via ChatGPT and Dataset Augmentation"*, arXiv preprint:2409.12599v1, 2024.

[12] David Grangier, Angelos Katharopoulos, Pierre Ablin, Awni Hannun, *"Need a Small Specialized Language Model? Plan Early!"*, arXiv preprint:2402.01093v2, 2024.

[13] Ben Fauber, *"Learning the Latent Rules of a Game from Data: A Chess Story"*, arXiv preprint:2410.02426v1, 2024.

[14] Wenyu Huang, Guancheng Zhou, Hongru Wang, Pavlos Vougiouklis, Mirella Lapata, Jeff Z. Pan, *"Less is More: Making Smaller Language Models Competent Subgraph  Retrievers for Multi-hop KGQA"*, arXiv preprint:2410.06121v1, 2024.

[15] Uri Shaham, Jonathan Herzig, Roee Aharoni, Idan Szpektor, Reut Tsarfaty, Matan Eyal, *"Multilingual Instruction Tuning With Just a Pinch of Multilinguality"*, arXiv preprint:2401.01854v4, 2024.

[16] Leonardo Ranaldi, Andrè Freitas, *"Self-Refine Instruction-Tuning for Aligning Reasoning in Language Models"*, arXiv preprint:2405.00402v1, 2024.

[17] Laurence Liang, *"Stacking Small Language Models for Generalizability"*, arXiv preprint:2410.15570v1, 2024.

[18] Yukang Xie, Chengyu Wang, Junbing Yan, Jiyong Zhou, Feiqi Deng, Jun Huang, *"Making Small Language Models Better Multi-task Learners with  Mixture-of-Task-Adapters"*, arXiv preprint:2309.11042v1, 2023.

[19] Zhanhui Zhou, Zhixuan Liu, Jie Liu, Zhichen Dong, Chao Yang, Yu Qiao, *"Weak-to-Strong Search: Align Large Language Models via Searching over  Small Language Models"*, arXiv preprint:2405.19262v3, 2024.

[20] Milan Bhan, Jean-Noel Vittaut, Nicolas Chesneau, Marie-Jeanne Lesot, *"Self-AMPLIFY: Improving Small Language Models with Self Post Hoc  Explanations"*, arXiv preprint:2402.12038v3, 2024.

[21] Shreyas Subramanian, Vikram Elango, Mecit Gungor, *"Small Language Models (SLMs) Can Still Pack a Punch: A survey"*, arXiv preprint:2501.05465v1, 2025.

[22] Haitao Li, Qingyao Ai, Jia Chen, Qian Dong, Zhijing Wu, Yiqun Liu, Chong Chen, Qi Tian, *"BLADE: Enhancing Black-box Large Language Models with Small  Domain-Specific Models"*, arXiv preprint:2403.18365v1, 2024.

[23] Nankai Lin, Yingwen Fu, Chuwei Chen, Ziyu Yang, Shengyi Jiang, *"LaoPLM: Pre-trained Language Models for Lao"*, arXiv preprint:2110.05896v3, 2021.

[24] Julien Khlaut, Corentin Dancette, Elodie Ferreres, Alaedine Bennani, Paul Hérent, Pierre Manceron, *"Efficient Medical Question Answering with Knowledge-Augmented Question  Generation"*, arXiv preprint:2405.14654v1, 2024.

[25] Yifei He, Alon Benhaim, Barun Patra, Praneetha Vaddamanu, Sanchit Ahuja, Parul Chopra, Vishrav Chaudhary, Han Zhao, Xia Song, *"Scaling Laws for Multilingual Language Models"*, arXiv preprint:2410.12883v2, 2024.

[26] Fan Zhang, Kebing Jin, Hankz Hankui Zhuo, *"Planning with Logical Graph-based Language Model for Instruction  Generation"*, arXiv preprint:2308.13782v2, 2023.

[27] Sia Gholami, Marwan Omar, *"Do Generative Large Language Models need billions of parameters?"*, arXiv preprint:2309.06589v1, 2023.

[28] Mitodru Niyogi, Arnab Bhattacharya, *"Paramanu: A Family of Novel Efficient Generative Foundation Language  Models for Indian Languages"*, arXiv preprint:2401.18034v2, 2024.

[29] François Remy, Pieter Delobelle, Bettina Berendt, Kris Demuynck, Thomas Demeester, *"Tik-to-Tok: Translating Language Models One Token at a Time: An  Embedding Initialization Strategy for Efficient Language Adaptation"*, arXiv preprint:2310.03477v1, 2023.

[30] Simon Kurz, Jian-Jia Chen, Lucie Flek, Zhixue Zhao, *"Investigating Language-Specific Calibration For Pruning Multilingual  Large Language Models"*, arXiv preprint:2408.14398v3, 2024.

[31] Guorui Zheng, Xidong Wang, Juhao Liang, Nuo Chen, Yuping Zheng, Benyou Wang, *"Efficiently Democratizing Medical LLMs for 50 Languages via a Mixture of  Language Family Experts"*, arXiv preprint:2410.10626v2, 2024.

[32] Samuel Cahyawijaya, *"LLM for Everyone: Representing the Underrepresented in Large Language  Models"*, arXiv preprint:2409.13897v1, 2024.

[33] Jianghao Lin, Xinyi Dai, Rong Shan, Bo Chen, Ruiming Tang, Yong Yu, Weinan Zhang, *"Large Language Models Make Sample-Efficient Recommender Systems"*, arXiv preprint:2406.02368v1, 2024.

[34] Atsuki Yamaguchi, Aline Villavicencio, Nikolaos Aletras, *"An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient  Language Model Inference"*, arXiv preprint:2402.10712v3, 2024.

[35] Jianhong Tu, Zhuohao Ni, Nicholas Crispino, Zihao Yu, Michael Bendersky, Beliz Gunel, Ruoxi Jia, Xin Liu, Lingjuan Lyu, Dawn Song, Chenguang Wang, *"MLAN: Language-Based Instruction Tuning Improves Zero-Shot  Generalization of Multimodal Large Language Models"*, arXiv preprint:2411.10557v2, 2024.

[36] Shaolei Zhang, Kehao Zhang, Qingkai Fang, Shoutao Guo, Yan Zhou, Xiaodong Liu, Yang Feng, *"BayLing 2: A Multilingual Large Language Model with Efficient Language  Alignment"*, arXiv preprint:2411.16300v3, 2024.

[37] Jiacheng Ye, Chengzu Li, Lingpeng Kong, Tao Yu, *"Generating Data for Symbolic Language with Large Language Models"*, arXiv preprint:2305.13917v1, 2023.

[38] Kelly Marchisio, Wei-Yin Ko, Alexandre Bérard, Théo Dehaze, Sebastian Ruder, *"Understanding and Mitigating Language Confusion in LLMs"*, arXiv preprint:2406.20052v2, 2024.

[39] Caoyun Fan, Wenqing Chen, Jidong Tian, Yitian Li, Hao He, Yaohui Jin, *"Improving the Out-Of-Distribution Generalization Capability of Language  Models: Counterfactually-Augmented Data is not Enough"*, arXiv preprint:2302.09345v1, 2023.

[40] Payal Bajaj, Chenyan Xiong, Guolin Ke, Xiaodong Liu, Di He, Saurabh Tiwary, Tie-Yan Liu, Paul Bennett, Xia Song, Jianfeng Gao, *"METRO: Efficient Denoising Pretraining of Large Scale Autoencoding  Language Models with Model Generated Signals"*, arXiv preprint:2204.06644v2, 2022.

[41] Wenye Lin, Yangning Li, Yifeng Ding, Hai-Tao Zheng, *"Tree-structured Auxiliary Online Knowledge Distillation"*, arXiv preprint:2208.10068v1, 2022.

[42] Shicheng Tan, Weng Lam Tam, Yuanchun Wang, Wenwen Gong, Yang Yang, Hongyin Tang, Keqing He, Jiahao Liu, Jingang Wang, Shu Zhao, Peng Zhang, Jie Tang, *"GKD: A General Knowledge Distillation Framework for Large-scale  Pre-trained Language Model"*, arXiv preprint:2306.06629v1, 2023.

[43] Jingxuan Wei, Linzhuang Sun, Yichong Leng, Xu Tan, Bihui Yu, Ruifeng Guo, *"Sentence-Level or Token-Level? A Comprehensive Study on Knowledge  Distillation"*, arXiv preprint:2404.14827v1, 2024.

[44] Zengkui Sun, Yijin Liu, Fandong Meng, Yufeng Chen, Jinan Xu, Jie Zhou, *"Warmup-Distill: Bridge the Distribution Mismatch between Teacher and  Student before Knowledge Distillation"*, arXiv preprint:2502.11766v1, 2025.

[45] Dongkyu Lee, Zhiliang Tian, Yingxiu Zhao, Ka Chun Cheung, Nevin L. Zhang, *"Hard Gate Knowledge Distillation -- Leverage Calibration for Robust and  Reliable Language Model"*, arXiv preprint:2210.12427v1, 2022.

[46] Hao Peng, Xin Lv, Yushi Bai, Zijun Yao, Jiajie Zhang, Lei Hou, Juanzi Li, *"Pre-training Distillation for Large Language Models: A Design Space  Exploration"*, arXiv preprint:2410.16215v1, 2024.

[47] Chenglong Wang, Yi Lu, Yongyu Mu, Yimin Hu, Tong Xiao, Jingbo Zhu, *"Improved Knowledge Distillation for Pre-trained Language Models via  Knowledge Selection"*, arXiv preprint:2302.00444v1, 2023.

[48] Ying Zhang, Ziheng Yang, Shufan Ji, *"MLKD-BERT: Multi-level Knowledge Distillation for Pre-trained Language  Models"*, arXiv preprint:2407.02775v1, 2024.

[49] Geondo Park, Gyeongman Kim, Eunho Yang, *"Distilling Linguistic Context for Language Model Compression"*, arXiv preprint:2109.08359v1, 2021.

[50] Mitchell A. Gordon, Kevin Duh, *"Distill, Adapt, Distill: Training Small, In-Domain Models for Neural  Machine Translation"*, arXiv preprint:2003.02877v3, 2020.

[51] Siyue Wu, Hongzhan Chen, Xiaojun Quan, Qifan Wang, Rui Wang, *"AD-KD: Attribution-Driven Knowledge Distillation for Language Model  Compression"*, arXiv preprint:2305.10010v1, 2023.

[52] Chen Liang, Haoming Jiang, Zheng Li, Xianfeng Tang, Bin Yin, Tuo Zhao, *"HomoDistil: Homotopic Task-Agnostic Distillation of Pre-trained  Transformers"*, arXiv preprint:2302.09632v1, 2023.

[53] Bowen Wu, Huan Zhang, Mengyuan Li, Zongsheng Wang, Qihang Feng, Junhong Huang, Baoxun Wang, *"Towards Non-task-specific Distillation of BERT via Sentence  Representation Approximation"*, arXiv preprint:2004.03097v1, 2020.

[54] Lakshmi Nair, *"CLIP-Embed-KD: Computationally Efficient Knowledge Distillation Using  Embeddings as Teachers"*, arXiv preprint:2404.06170v1, 2024.

[55] Flavio Di Palo, Prateek Singhi, Bilal Fadlallah, *"Performance-Guided LLM Knowledge Distillation for Efficient Text  Classification at Scale"*, arXiv preprint:2411.05045v1, 2024.

[56] Karthik S. Vedula, Annika Gupta, Akshay Swaminathan, Ivan Lopez, Suhana Bedi, Nigam H. Shah, *"Distilling Large Language Models for Efficient Clinical Information  Extraction"*, arXiv preprint:2501.00031v1, 2024.

[57] Siqi Sun, Zhe Gan, Yu Cheng, Yuwei Fang, Shuohang Wang, Jingjing Liu, *"Contrastive Distillation on Intermediate Representations for Language  Model Compression"*, arXiv preprint:2009.14167v1, 2020.

[58] Siqi Sun, Yu Cheng, Zhe Gan, Jingjing Liu, *"Patient Knowledge Distillation for BERT Model Compression"*, arXiv preprint:1908.09355v1, 2019.

[59] James O' Neill, Sourav Dutta, Haytham Assem, *"Deep Neural Compression Via Concurrent Pruning and Self-Distillation"*, arXiv preprint:2109.15014v1, 2021.

[60] Fahimeh Saleh, Wray Buntine, Gholamreza Haffari, *"Collective Wisdom: Improving Low-resource Neural Machine Translation  using Adaptive Knowledge Distillation"*, arXiv preprint:2010.05445v1, 2020.

[61] Jiaheng Liu, Chenchen Zhang, Jinyang Guo, Yuanxing Zhang, Haoran Que, Ken Deng, Zhiqi Bai, Jie Liu, Ge Zhang, Jiakai Wang, Yanan Wu, Congnan Liu, Wenbo Su, Jiamang Wang, Lin Qu, Bo Zheng, *"DDK: Distilling Domain Knowledge for Efficient Large Language Models"*, arXiv preprint:2407.16154v1, 2024.

[62] Mingsheng Li, Lin Zhang, Mingzhen Zhu, Zilong Huang, Gang Yu, Jiayuan Fan, Tao Chen, *"Lightweight Model Pre-training via Language Guided Knowledge  Distillation"*, arXiv preprint:2406.11689v1, 2024.

[63] Gustavo Aguilar, Yuan Ling, Yu Zhang, Benjamin Yao, Xing Fan, Chenlei Guo, *"Knowledge Distillation from Internal Representations"*, arXiv preprint:1910.03723v2, 2019.

[64] Jan Christian Blaise Cruz, Alham Fikri Aji, *"Extracting General-use Transformers for Low-resource Languages via  Knowledge Distillation"*, arXiv preprint:2501.12660v1, 2025.

[65] Umang Gupta, Jwala Dhamala, Varun Kumar, Apurv Verma, Yada Pruksachatkun, Satyapriya Krishna, Rahul Gupta, Kai-Wei Chang, Greg Ver Steeg, Aram Galstyan, *"Mitigating Gender Bias in Distilled Language Models via Counterfactual  Role Reversal"*, arXiv preprint:2203.12574v1, 2022.

[66] Mun-Hak Lee, Joon-Hyuk Chang, *"Knowledge distillation from language model to acoustic model: a  hierarchical multi-task learning approach"*, arXiv preprint:2110.10429v1, 2021.

[67] Haojie Pan, Chengyu Wang, Minghui Qiu, Yichang Zhang, Yaliang Li, Jun Huang, *"Meta-KD: A Meta Knowledge Distillation Framework for Language Model  Compression across Domains"*, arXiv preprint:2012.01266v2, 2020.

[68] Ziqing Yang, Yiming Cui, Zhigang Chen, *"TextPruner: A Model Pruning Toolkit for Pre-Trained Language Models"*, arXiv preprint:2203.15996v1, 2022.

[69] Zhewei Yao, Xiaoxia Wu, Linjian Ma, Sheng Shen, Kurt Keutzer, Michael W. Mahoney, Yuxiong He, *"LEAP: Learnable Pruning for Transformer-based Models"*, arXiv preprint:2105.14636v2, 2021.

[70] Seungcheol Park, Hojun Choi, U Kang, *"Accurate Retraining-free Pruning for Pretrained Encoder-based Language  Models"*, arXiv preprint:2308.03449v2, 2023.

[71] Jaeseong Lee, seung-won hwang, Aurick Qiao, Daniel F Campos, Zhewei Yao, Yuxiong He, *"STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning"*, arXiv preprint:2409.06211v1, 2024.

[72] Gui Ling, Ziyang Wang, Yuliang Yan, Qingwen Liu, *"SlimGPT: Layer-wise Structured Pruning for Large Language Models"*, arXiv preprint:2412.18110v1, 2024.

[73] Xiaodong Chen, Yuxuan Hu, Xiaokang Zhang, Yanling Wang, Cuiping Li, Hong Chen, Jing Zhang, *"P$^2$ Law: Scaling Law for Post-Training After Model Pruning"*, arXiv preprint:2411.10272v2, 2024.

[74] Guanchen Li, Xiandong Zhao, Lian Liu, Zeping Li, Dong Li, Lu Tian, Jie He, Ashish Sirasao, Emad Barsoum, *"Enhancing One-shot Pruned Pre-trained Language Models through  Sparse-Dense-Sparse Mechanism"*, arXiv preprint:2408.10473v1, 2024.

[75] Samarth N Ramesh, Zhixue Zhao, *"Efficient Pruning of Text-to-Image Models: Insights from Pruning Stable  Diffusion"*, arXiv preprint:2411.15113v1, 2024.

[76] Siyu Ren, Kenny Q. Zhu, *"Pruning Pre-trained Language Models with Principled Importance and  Self-regularization"*, arXiv preprint:2305.12394v1, 2023.

[77] Ziqing Yang, Yiming Cui, Xin Yao, Shijin Wang, *"Gradient-based Intra-attention Pruning on Pre-trained Language Models"*, arXiv preprint:2212.07634v2, 2022.

[78] Michael Santacroce, Zixin Wen, Yelong Shen, Yuanzhi Li, *"What Matters In The Structured Pruning of Generative Language Models?"*, arXiv preprint:2302.03773v1, 2023.

[79] Minsik Cho, Saurabh Adya, Devang Naik, *"PDP: Parameter-free Differentiable Pruning is All You Need"*, arXiv preprint:2305.11203v3, 2023.

[80] Peijie Dong, Lujun Li, Zhenheng Tang, Xiang Liu, Xinglin Pan, Qiang Wang, Xiaowen Chu, *"Pruner-Zero: Evolving Symbolic Pruning Metric from scratch for Large  Language Models"*, arXiv preprint:2406.02924v1, 2024.

[81] Hongrong Cheng, Miao Zhang, Javen Qinfeng Shi, *"A Survey on Deep Neural Network Pruning-Taxonomy, Comparison, Analysis,  and Recommendations"*, arXiv preprint:2308.06767v2, 2023.

[82] Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu, *"Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy  for Language Models"*, arXiv preprint:2310.13191v3, 2023.

[83] Runxin Xu, Fuli Luo, Chengyu Wang, Baobao Chang, Jun Huang, Songfang Huang, Fei Huang, *"From Dense to Sparse: Contrastive Pruning for Better Pre-trained  Language Model Compression"*, arXiv preprint:2112.07198v1, 2021.

[84] Eldar Kurtic, Torsten Hoefler, Dan Alistarh, *"How to Prune Your Language Model: Recovering Accuracy on the "Sparsity  May Cry'' Benchmark"*, arXiv preprint:2312.13547v1, 2023.

[85] Yanyue Xie, Zhi Zhang, Ding Zhou, Cong Xie, Ziang Song, Xin Liu, Yanzhi Wang, Xue Lin, An Xu, *"MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the  Hints from Its Router"*, arXiv preprint:2410.12013v1, 2024.

[86] Bo-Kyeong Kim, Geonmin Kim, Tae-Ho Kim, Thibault Castells, Shinkook Choi, Junho Shin, Hyoung-Kyu Song, *"Shortened LLaMA: Depth Pruning for Large Language Models with Comparison  of Retraining Methods"*, arXiv preprint:2402.02834v2, 2024.

[87] Hanyu Hu, Pengxiang Zhao, Ping Li, Yi Zheng, Zhefeng Wang, Xiaoming Yuan, *"FASP: Fast and Accurate Structured Pruning of Large Language Models"*, arXiv preprint:2501.09412v1, 2025.

[88] Kyuhong Shim, Iksoo Choi, Wonyong Sung, Jungwook Choi, *"Layer-wise Pruning of Transformer Attention Heads for Efficient Language  Modeling"*, arXiv preprint:2110.03252v1, 2021.

[89] Jianwei Li, Yijun Dong, Qi Lei, *"Greedy Output Approximation: Towards Efficient Structured Pruning for  LLMs Without Retraining"*, arXiv preprint:2407.19126v1, 2024.

[90] Dongjun Park, Geung-Hee Lee, *"Structured Pattern Pruning Using Regularization"*, arXiv preprint:2109.08814v1, 2021.

[91] Przemyslaw Joniak, Akiko Aizawa, *"Gender Biases and Where to Find Them: Exploring Gender Bias in  Pre-Trained Transformer-based Language Models Using Movement Pruning"*, arXiv preprint:2207.02463v1, 2022.

[92] Shuqi Liu, Bowei He, Han Wu, Linqi Song, *"OPTISHEAR: Towards Efficient and Adaptive Pruning of Large Language  Models via Evolutionary Optimization"*, arXiv preprint:2502.10735v1, 2025.

[93] Zachary Ankner, Cody Blakeney, Kartik Sreenivasan, Max Marion, Matthew L. Leavitt, Mansheej Paul, *"Perplexed by Perplexity: Perplexity-Based Data Pruning With Small  Reference Models"*, arXiv preprint:2405.20541v1, 2024.

[94] James O' Neill, Sourav Dutta, Haytham Assem, *"Aligned Weight Regularizers for Pruning Pretrained Neural Networks"*, arXiv preprint:2204.01385v2, 2022.

[95] Lei Lu, Zhepeng Wang, Runxue Bao, Mengbing Wang, Fangyi Li, Yawen Wu, Weiwen Jiang, Jie Xu, Yanzhi Wang, Shangqian Gao, *"All-in-One Tuning and Structural Pruning for Domain-Specific LLMs"*, arXiv preprint:2412.14426v2, 2024.

[96] Mingxuan Zhang, Yan Sun, Faming Liang, *"Magnitude Pruning of Large Pretrained Transformer Models with a Mixture  Gaussian Prior"*, arXiv preprint:2411.00969v1, 2024.

[97] Song Guo, Jiahang Xu, Li Lyna Zhang, Mao Yang, *"Compresso: Structured Pruning with Collaborative Prompting Learns  Compact Large Language Models"*, arXiv preprint:2310.05015v2, 2023.

[98] Vladimír Boža, *"Fast and Effective Weight Update for Pruned Large Language Models"*, arXiv preprint:2401.02938v2, 2024.

[99] Shangyu Wu, Hongchao Du, Ying Xiong, Shuai Chen, Tei-wei Kuo, Nan Guan, Chun Jason Xue, *"EvoP: Robust LLM Inference via Evolutionary Pruning"*, arXiv preprint:2502.14910v1, 2025.

[100] Binh-Nguyen Nguyen, Yang He, *"Swift Cross-Dataset Pruning: Enhancing Fine-Tuning Efficiency in Natural  Language Understanding"*, arXiv preprint:2501.02432v1, 2025.

[101] Boyao Wang, Rui Pan, Shizhe Diao, Xingyuan Pan, Jipeng Zhang, Renjie Pi, Tong Zhang, *"Adapt-Pruner: Adaptive Structural Pruning for Efficient Small Language  Model Training"*, arXiv preprint:2502.03460v1, 2025.

[102] Zihuai Xu, Yang Xu, Hongli Xu, Yunming Liao, Zhiwei Yao, Zuan Xie, *"Lightweight and Post-Training Structured Pruning for On-Device Large  Lanaguage Models"*, arXiv preprint:2501.15255v1, 2025.

[103] Longguang Zhong, Fanqi Wan, Ruijun Chen, Xiaojun Quan, Liangzhi Li, *"BlockPruner: Fine-grained Pruning for Large Language Models"*, arXiv preprint:2406.10594v3, 2024.

[104] Jun Liu, Zhenglun Kong, Pu Zhao, Changdi Yang, Hao Tang, Xuan Shen, Geng Yuan, Wei Niu, Wenbin Zhang, Xue Lin, Dong Huang, Yanzhi Wang, *"Toward Adaptive Large Language Models Structured Pruning via  Hybrid-grained Weight Importance Assessment"*, arXiv preprint:2403.10799v5, 2024.

[105] Yongqi An, Xu Zhao, Tao Yu, Ming Tang, Jinqiao Wang, *"Fluctuation-based Adaptive Structured Pruning for Large Language Models"*, arXiv preprint:2312.11983v1, 2023.

[106] Guangji Bai, Yijiang Li, Zilinghan Li, Liang Zhao, Kibaek Kim, *"FedSpaLLM: Federated Pruning of Large Language Models"*, arXiv preprint:2410.14852v2, 2024.

[107] Yuan Gao, Zujing Liu, Weizhong Zhang, Bo Du, Gui-Song Xia, *"Bypass Back-propagation: Optimization-based Structural Pruning for Large  Language Models via Policy Gradient"*, arXiv preprint:2406.10576v2, 2024.

[108] Anushka Shelke, Riya Savant, Raviraj Joshi, *"Towards Building Efficient Sentence BERT Models using Layer Pruning"*, arXiv preprint:2409.14168v1, 2024.

[109] Matteo Farina, Massimiliano Mancini, Elia Cunegatti, Gaowen Liu, Giovanni Iacca, Elisa Ricci, *"MULTIFLOW: Shifting Towards Task-Agnostic Vision-Language Pruning"*, arXiv preprint:2404.05621v1, 2024.

[110] Yunshui Li, Junhao Liu, Chengming Li, Min Yang, *"Self-Distillation with Meta Learning for Knowledge Graph Completion"*, arXiv preprint:2305.12209v1, 2023.

[111] Shengrui Li, Junzhe Chen, Xueting Han, Jing Bai, *"NutePrune: Efficient Progressive Pruning with Numerous Teachers for  Large Language Models"*, arXiv preprint:2402.09773v2, 2024.

[112] Weizhong Huang, Yuxin Zhang, Xiawu Zheng, Fei Chao, Rongrong Ji, *"Towards Efficient Automatic Self-Pruning of Large Language Models"*, arXiv preprint:2502.14413v1, 2025.

[113] Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter, *"A Simple and Effective Pruning Approach for Large Language Models"*, arXiv preprint:2306.11695v3, 2023.

[114] Dongkuan Xu, Ian E. H. Yen, Jinxi Zhao, Zhibin Xiao, *"Rethinking Network Pruning -- under the Pre-train and Fine-tune Paradigm"*, arXiv preprint:2104.08682v2, 2021.

[115] Bowen Shen, Zheng Lin, Daren Zha, Wei Liu, Jian Luan, Bin Wang, Weiping Wang, *"Pruning Large Language Models to Intra-module Low-rank Architecture with  Transitional Activations"*, arXiv preprint:2407.05690v1, 2024.

[116] Wenyuan Liu, Xindian Ma, Peng Zhang, Yan Wang, *"CrossQuant: A Post-Training Quantization Method with Smaller  Quantization Kernel for Precise Large Language Model Compression"*, arXiv preprint:2410.07505v1, 2024.

[117] James O' Neill, Sourav Dutta, *"Self-Distilled Quantization: Achieving High Compression Rates in  Transformer-Based Language Models"*, arXiv preprint:2307.05972v1, 2023.

[118] Zihan Zhao, Yuncong Liu, Lu Chen, Qi Liu, Rao Ma, Kai Yu, *"An Investigation on Different Underlying Quantization Schemes for  Pre-trained Language Models"*, arXiv preprint:2010.07109v1, 2020.

[119] Jing Jin, Cai Liang, Tiancheng Wu, Liqin Zou, Zhiliang Gan, *"KDLSQ-BERT: A Quantized Bert Combining Knowledge Distillation with  Learned Step Size Quantization"*, arXiv preprint:2101.05938v1, 2021.

[120] Irina Proskurina, Luc Brun, Guillaume Metzler, Julien Velcin, *"When Quantization Affects Confidence of Large Language Models?"*, arXiv preprint:2405.00632v1, 2024.

[121] Yuxuan Yue, Zhihang Yuan, Haojie Duanmu, Sifan Zhou, Jianlong Wu, Liqiang Nie, *"WKVQuant: Quantizing Weight and Key/Value Cache for Large Language  Models Gains More"*, arXiv preprint:2402.12065v2, 2024.

[122] Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, Tuo Zhao, *"LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models"*, arXiv preprint:2310.08659v4, 2023.

[123] Baisong Li, Xingwang Wang, Haixiao Xu, *"AWEQ: Post-Training Quantization with Activation-Weight Equalization for  Large Language Models"*, arXiv preprint:2311.01305v3, 2023.

[124] Neal Lawton, Aishwarya Padmakumar, Judith Gaspers, Jack FitzGerald, Anoop Kumar, Greg Ver Steeg, Aram Galstyan, *"QuAILoRA: Quantization-Aware Initialization for LoRA"*, arXiv preprint:2410.14713v1, 2024.

[125] Jingjing Xie, Yuxin Zhang, Mingbao Lin, Liujuan Cao, Rongrong Ji, *"Advancing Multimodal Large Language Models with Quantization-Aware Scale  Learning for Efficient Adaptation"*, arXiv preprint:2408.03735v1, 2024.

[126] Junhao Xu, Xie Chen, Shoukang Hu, Jianwei Yu, Xunying Liu, Helen Meng, *"Low-bit Quantization of Recurrent Neural Network Language Models Using  Alternating Direction Methods of Multipliers"*, arXiv preprint:2111.14836v1, 2021.

[127] Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, Vikas Chandra, *"LLM-QAT: Data-Free Quantization Aware Training for Large Language Models"*, arXiv preprint:2305.17888v1, 2023.

[128] Zeyu Cao, Cheng Zhang, Pedro Gimenes, Jianqiao Lu, Jianyi Cheng, Yiren Zhao, *"Scaling Laws for Mixed quantization in Large Language Models"*, arXiv preprint:2410.06722v1, 2024.

[129] Jahid Hasan, *"Optimizing Large Language Models through Quantization: A Comparative  Analysis of PTQ and QAT Techniques"*, arXiv preprint:2411.06084v1, 2024.

[130] Minseop Park, Jaeseong You, Markus Nagel, Simyung Chang, *"Quadapter: Adapter for GPT-2 Quantization"*, arXiv preprint:2211.16912v1, 2022.

[131] Aniruddha Nrusimha, Mayank Mishra, Naigang Wang, Dan Alistarh, Rameswar Panda, Yoon Kim, *"Mitigating the Impact of Outlier Channels for Language Model  Quantization with Activation Regularization"*, arXiv preprint:2404.03605v2, 2024.

[132] Renren Jin, Jiangcun Du, Wuwei Huang, Wei Liu, Jian Luan, Bin Wang, Deyi Xiong, *"A Comprehensive Evaluation of Quantization Strategies for Large Language  Models"*, arXiv preprint:2402.16775v2, 2024.

[133] Yifei Gao, Jie Ou, Lei Wang, Yuting Xiao, Zhiyuan Xiang, Ruiting Dai, Jun Cheng, *"Compensate Quantization Errors: Make Weights Hierarchical to Compensate  Each Other"*, arXiv preprint:2406.16299v1, 2024.

[134] Yuji Chai, John Gkountouras, Glenn G. Ko, David Brooks, Gu-Yeon Wei, *"INT2.1: Towards Fine-Tunable Quantized Large Language Models with Error  Correction through Low-Rank Adaptation"*, arXiv preprint:2306.08162v1, 2023.

[135] Zhuocheng Gong, Jiahao Liu, Qifan Wang, Yang Yang, Jingang Wang, Wei Wu, Yunsen Xian, Dongyan Zhao, Rui Yan, *"PreQuant: A Task-agnostic Quantization Approach for Pre-trained Language  Models"*, arXiv preprint:2306.00014v1, 2023.

[136] Xin Ding, Shijie Cao, Ting Cao, Zhibo Chen, *"Dissecting Bit-Level Scaling Laws in Quantizing Vision Generative Models"*, arXiv preprint:2501.06218v1, 2025.

[137] Jaeseong You, Minseop Park, Kyunggeun Lee, Seokjun An, Chirag Patel, Markus Nage, *"How to Parameterize Asymmetric Quantization Ranges for  Quantization-Aware Training"*, arXiv preprint:2404.16898v1, 2024.

[138] Junhao Xu, Jianwei Yu, Shoukang Hu, Xunying Liu, Helen Meng, *"Mixed Precision Low-bit Quantization of Neural Network Language Models  for Speech Recognition"*, arXiv preprint:2112.11438v1, 2021.

[139] Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, Ping Luo, *"OmniQuant: Omnidirectionally Calibrated Quantization for Large Language  Models"*, arXiv preprint:2308.13137v3, 2023.

[140] Zhikai Li, Xuewen Liu, Jing Zhang, Qingyi Gu, *"RepQuant: Towards Accurate Post-Training Quantization of Large  Transformer Models via Scale Reparameterization"*, arXiv preprint:2402.05628v1, 2024.

[141] Mengzhao Chen, Yi Liu, Jiahao Wang, Yi Bin, Wenqi Shao, Ping Luo, *"PrefixQuant: Eliminating Outliers by Prefixed Tokens for Large Language  Models Quantization"*, arXiv preprint:2410.05265v2, 2024.

[142] Weibo Zhao, Yubin Shi, Xinyu Lyu, Wanchen Sui, Shen Li, Yong Li, *"ASER: Activation Smoothing and Error Reconstruction for Large Language  Model Quantization"*, arXiv preprint:2411.07762v2, 2024.

[143] Peiyu Liu, Zikang Liu, Ze-Feng Gao, Dawei Gao, Wayne Xin Zhao, Yaliang Li, Bolin Ding, Ji-Rong Wen, *"Do Emergent Abilities Exist in Quantized Large Language Models: An  Empirical Study"*, arXiv preprint:2307.08072v2, 2023.

[144] Zihan Chen, Bike Xie, Jundong Li, Cong Shen, *"Channel-Wise Mixed-Precision Quantization for Large Language Models"*, arXiv preprint:2410.13056v3, 2024.

[145] Kelly Marchisio, Saurabh Dash, Hongyu Chen, Dennis Aumiller, Ahmet Üstün, Sara Hooker, Sebastian Ruder, *"How Does Quantization Affect Multilingual LLMs?"*, arXiv preprint:2407.03211v2, 2024.

[146] Yifei Gao, Jie Ou, Lei Wang, Fanhua Shang, Jaji Wu, Jun Cheng, *"Compensate Quantization Errors+: Quantized Models Are Inquisitive  Learners"*, arXiv preprint:2407.15508v2, 2024.

[147] Pengxiang Zhao, Xiaoming Yuan, *"GANQ: GPU-Adaptive Non-Uniform Quantization for Large Language Models"*, arXiv preprint:2501.12956v2, 2025.

[148] Noga Bar, Raja Giryes, *"ZOQO: Zero-Order Quantized Optimization"*, arXiv preprint:2501.06736v1, 2025.

[149] Cheng Chen, Christina Giannoula, Andreas Moshovos, *"Low-Bitwidth Floating Point Quantization for Efficient High-Quality  Diffusion Models"*, arXiv preprint:2408.06995v1, 2024.

[150] Seyed Parsa Neshaei, Yasaman Boreshban, Gholamreza Ghassem-Sani, Seyed Abolghasem Mirroshandel, *"The Impact of Quantization on the Robustness of Transformer-based Text  Classifiers"*, arXiv preprint:2403.05365v1, 2024.

[151] Baohao Liao, Christian Herold, Shahram Khadivi, Christof Monz, *"ApiQ: Finetuning of 2-Bit Quantized Large Language Model"*, arXiv preprint:2402.05147v3, 2024.

[152] Chao Zeng, Songwei Liu, Yusheng Xie, Hong Liu, Xiaojian Wang, Miao Wei, Shu Yang, Fangmin Chen, Xing Mei, *"ABQ-LLM: Arbitrary-Bit Quantized Inference Acceleration for Large  Language Models"*, arXiv preprint:2408.08554v2, 2024.

[153] Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort, *"Understanding and Overcoming the Challenges of Efficient Transformer  Quantization"*, arXiv preprint:2109.12948v1, 2021.

[154] Zhuocheng Gong, Jiahao Liu, Jingang Wang, Xunliang Cai, Dongyan Zhao, Rui Yan, *"What Makes Quantization for Large Language Models Hard? An Empirical  Study from the Lens of Perturbation"*, arXiv preprint:2403.06408v1, 2024.

[155] Pingzhi Li, Xiaolong Jin, Yu Cheng, Tianlong Chen, *"Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark"*, arXiv preprint:2406.08155v1, 2024.

[156] Enkhbold Nyamsuren, *"Evaluating Quantized Large Language Models for Code Generation on  Low-Resource Language Benchmarks"*, arXiv preprint:2410.14766v1, 2024.

[157] Ting Hu, Christoph Meinel, Haojin Yang, *"Empirical Evaluation of Post-Training Quantization Methods for Language  Tasks"*, arXiv preprint:2210.16621v1, 2022.

[158] Nilesh Prasad Pandey, Marios Fournarakis, Chirag Patel, Markus Nagel, *"Softmax Bias Correction for Quantized Generative Models"*, arXiv preprint:2309.01729v1, 2023.

[159] Tianyi Zhang, Anshumali Shrivastava, *"LeanQuant: Accurate and Scalable Large Language Model Quantization with  Loss-error-aware Grid"*, arXiv preprint:2407.10032v2, 2024.

[160] Xu Ouyang, Tao Ge, Thomas Hartvigsen, Zhisong Zhang, Haitao Mi, Dong Yu, *"Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for  Quantized LLMs with 100T Training Tokens"*, arXiv preprint:2411.17691v2, 2024.

[161] Khasmamad Shabanovi, Lukas Wiest, Vladimir Golkov, Daniel Cremers, Thomas Pfeil, *"Interactions Across Blocks in Post-Training Quantization of Large  Language Models"*, arXiv preprint:2411.03934v1, 2024.

[162] Hyesung Jeon, Yulhwa Kim, Jae-joon Kim, *"L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large  Language Models"*, arXiv preprint:2402.04902v5, 2024.

[163] Mengzhao Chen, Wenqi Shao, Peng Xu, Jiahao Wang, Peng Gao, Kaipeng Zhang, Ping Luo, *"EfficientQAT: Efficient Quantization-Aware Training for Large Language  Models"*, arXiv preprint:2407.11062v2, 2024.

[164] Riccardo Bravin, Massimo Pavan, Hazem Hesham Yousef Shalby, Fabrizio Pittorino, Manuel Roveri, *"EmbBERT-Q: Breaking Memory Barriers in Embedded NLP"*, arXiv preprint:2502.10001v1, 2025.

[165] Himmet Toprak Kesgin, Muzaffer Kaan Yuce, Mehmet Fatih Amasyali, *"Developing and Evaluating Tiny to Medium-Sized Turkish BERT Models"*, arXiv preprint:2307.14134v1, 2023.

[166] Michiel Kamphuis, *"Tiny-Toxic-Detector: A compact transformer-based model for toxic content  detection"*, arXiv preprint:2409.02114v1, 2024.

[167] Yehui Tang, Fangcheng Liu, Yunsheng Ni, Yuchuan Tian, Zheyuan Bai, Yi-Qi Hu, Sichao Liu, Shangling Jui, Kai Han, Yunhe Wang, *"Rethinking Optimization and Architecture for Tiny Language Models"*, arXiv preprint:2402.02791v2, 2024.

[168] Jiayi Wu, Hao Sun, Hengyi Cai, Lixin Su, Shuaiqiang Wang, Dawei Yin, Xiang Li, Ming Gao, *"Cross-model Control: Improving Multiple Large Language Models in  One-time Training"*, arXiv preprint:2410.17599v1, 2024.

[169] Mark Bajo, Haruka Fukukawa, Ryuji Morita, Yuma Ogasawara, *"Efficient Adaptation of Multilingual Models for Japanese ASR"*, arXiv preprint:2412.10705v1, 2024.

[170] Tanmay Sen, Ansuman Das, Mrinmay Sen, *"HateTinyLLM : Hate Speech Detection Using Tiny Large Language Models"*, arXiv preprint:2405.01577v1, 2024.

[171] Dylan Hillier, Leon Guertler, Cheston Tan, Palaash Agrawal, Chen Ruirui, Bobby Cheng, *"Super Tiny Language Models"*, arXiv preprint:2405.14159v2, 2024.

[172] Prabhu Kaliamoorthi, Aditya Siddhant, Edward Li, Melvin Johnson, *"Distilling Large Language Models into Tiny and Effective Students using  pQRNN"*, arXiv preprint:2101.08890v1, 2021.

[173] Ke Yang, Volodymyr Kindratenko, ChengXiang Zhai, *"TinyHelen's First Curriculum: Training and Evaluating Tiny Language  Models in a Simpler Language Environment"*, arXiv preprint:2501.00522v1, 2024.

[174] Zebin Yang, Renze Chen, Taiqiang Wu, Ngai Wong, Yun Liang, Runsheng Wang, Ru Huang, Meng Li, *"MCUBERT: Memory-Efficient BERT Inference on Commodity Microcontrollers"*, arXiv preprint:2410.17957v1, 2024.

[175] Peter Belcak, Roger Wattenhofer, *"Tiny Transformers Excel at Sentence Compression"*, arXiv preprint:2410.23510v1, 2024.

[176] Zhe Cao, Zhi Qu, Hidetaka Kamigaito, Taro Watanabe, *"Exploring Intrinsic Language-specific Subspaces in Fine-tuning  Multilingual Neural Machine Translation"*, arXiv preprint:2409.05224v1, 2024.

[177] Zhongzhi Yu, Yonggan Fu, Jiayi Yuan, Haoran You, Yingyan Lin, *"NetBooster: Empowering Tiny Deep Learning By Standing on the Shoulders  of Deep Giants"*, arXiv preprint:2306.13586v1, 2023.

[178] Xinrun Du, Zhouliang Yu, Songyang Gao, Ding Pan, Yuyang Cheng, Ziyang Ma, Ruibin Yuan, Xingwei Qu, Jiaheng Liu, Tianyu Zheng, Xinchen Luo, Guorui Zhou, Wenhu Chen, Ge Zhang, *"Chinese Tiny LLM: Pretraining a Chinese-Centric Large Language Model"*, arXiv preprint:2404.04167v5, 2024.

[179] Adrián Bazaga, Pietro Liò, Gos Micklem, *"Language Model Knowledge Distillation for Efficient Question Answering  in Spanish"*, arXiv preprint:2312.04193v2, 2023.

[180] Gabrielle Cohn, Rishika Agarwal, Deepanshu Gupta, Siddharth Patwardhan, *"EELBERT: Tiny Models through Dynamic Embeddings"*, arXiv preprint:2310.20144v1, 2023.

[181] Nicholas Kluge Corrêa, Sophia Falk, Shiza Fatimah, Aniket Sen, Nythamar de Oliveira, *"TeenyTinyLlama: open-source tiny language models trained in Brazilian  Portuguese"*, arXiv preprint:2401.16640v3, 2024.

[182] Weiyue Su, Xuyi Chen, Shikun Feng, Jiaxiang Liu, Weixin Liu, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang, *"ERNIE-Tiny : A Progressive Distillation Framework for Pretrained  Transformer Compression"*, arXiv preprint:2106.02241v1, 2021.

[183] Qingwen Lin, Boyan Xu, Zhengting Huang, Ruichu Cai, *"From Large to Tiny: Distilling and Refining Mathematical Expertise for  Math Word Problems with Weakly Supervision"*, arXiv preprint:2403.14390v1, 2024.

[184] Shaikat Galib, Shanshan Wang, Guanshuo Xu, Pascal Pfeiffer, Ryan Chesler, Mark Landry, Sri Satish Ambati, *"H2OVL-Mississippi Vision Language Models Technical Report"*, arXiv preprint:2410.13611v1, 2024.

[185] Denis Tarasov, Kumar Shridhar, *"Distilling LLMs' Decomposition Abilities into Compact Language Models"*, arXiv preprint:2402.01812v1, 2024.

[186] Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, Le Sun, *"ToolAlpaca: Generalized Tool Learning for Language Models with 3000  Simulated Cases"*, arXiv preprint:2306.05301v2, 2023.

[187] Vincent Micheli, Martin d'Hoffschmidt, François Fleuret, *"On the importance of pre-training data volume for compact language  models"*, arXiv preprint:2010.03813v2, 2020.

[188] Christophe Servan, Sahar Ghannay, Sophie Rosset, *"mALBERT: Is a Compact Multilingual BERT Model Still Worth It?"*, arXiv preprint:2403.18338v1, 2024.

[189] Anthony Chen, Panupong Pasupat, Sameer Singh, Hongrae Lee, Kelvin Guu, *"PURR: Efficiently Editing Language Model Hallucinations by Denoising  Language Model Corruptions"*, arXiv preprint:2305.14908v1, 2023.

[190] Amirreza Esmaeili, Iman Saberi, Fatemeh H. Fard, *"Empirical Studies of Parameter Efficient Methods for Large Language  Models of Code and Knowledge Transfer to R"*, arXiv preprint:2405.01553v2, 2024.

[191] Zheyu Zhang, Han Yang, Bolei Ma, David Rügamer, Ercong Nie, *"Baby's CoThought: Leveraging Large Language Models for Enhanced  Reasoning in Compact Models"*, arXiv preprint:2308.01684v2, 2023.

[192] Rabeeh Karimi Mahabadi, James Henderson, Sebastian Ruder, *"Compacter: Efficient Low-Rank Hypercomplex Adapter Layers"*, arXiv preprint:2106.04647v2, 2021.

[193] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu, *"TinyLlama: An Open-Source Small Language Model"*, arXiv preprint:2401.02385v2, 2024.

[194] Amit Kumar Jaiswal, Haiming Liu, *"Lightweight Adaptation of Neural Language Models via Subspace Embedding"*, arXiv preprint:2308.08688v1, 2023.

[195] Irina Proskurina, Guillaume Metzler, Julien Velcin, *"Mini Minds: Exploring Bebeshka and Zlata Baby Models"*, arXiv preprint:2311.03216v1, 2023.

[196] Mohammadmahdi Nouriborji, Omid Rohanian, Samaneh Kouchaki, David A. Clifton, *"MiniALBERT: Model Distillation via Parameter-Efficient Recursive  Transformers"*, arXiv preprint:2210.06425v2, 2022.

[197] Shahriar Golchin, Mihai Surdeanu, Nazgol Tavabi, Ata Kiapour, *"A Compact Pretraining Approach for Neural Language Models"*, arXiv preprint:2208.12367v2, 2022.

[198] Bhargav Shandilya, Alexis Palmer, *"Boosting the Capabilities of Compact Models in Low-Data Contexts with  Large Language Models and Retrieval-Augmented Generation"*, arXiv preprint:2410.00387v1, 2024.

[199] Ashutosh Sathe, Sunita Sarawagi, *"Efficient Training of Language Models with Compact and Consistent Next  Token Distributions"*, arXiv preprint:2407.02819v1, 2024.

[200] Iulia Turc, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, *"Well-Read Students Learn Better: On the Importance of Pre-training  Compact Models"*, arXiv preprint:1908.08962v2, 2019.

[201] Wenxiao Wang, Wei Chen, Yicong Luo, Yongliu Long, Zhengkai Lin, Liye Zhang, Binbin Lin, Deng Cai, Xiaofei He, *"Model Compression and Efficient Inference for Large Language Models: A  Survey"*, arXiv preprint:2402.09748v1, 2024.

[202] Alexis Chevalier, Alexander Wettig, Anirudh Ajith, Danqi Chen, *"Adapting Language Models to Compress Contexts"*, arXiv preprint:2305.14788v2, 2023.

[203] Mukul Gagrani, Raghavv Goel, Wonseok Jeon, Junyoung Park, Mingu Lee, Christopher Lott, *"On Speculative Decoding for Multimodal Large Language Models"*, arXiv preprint:2404.08856v1, 2024.

[204] Richard Zhuang, Tianhao Wu, Zhaojin Wen, Andrew Li, Jiantao Jiao, Kannan Ramchandran, *"EmbedLLM: Learning Compact Representations of Large Language Models"*, arXiv preprint:2410.02223v2, 2024.

[205] Yaping Chai, Haoran Xie, Joe S. Qin, *"Text Data Augmentation for Large Language Models: A Comprehensive Survey  of Methods, Challenges, and Opportunities"*, arXiv preprint:2501.18845v1, 2025.

[206] Catherine Yeh, Donghao Ren, Yannick Assogba, Dominik Moritz, Fred Hohman, *"Exploring Empty Spaces: Human-in-the-Loop Data Augmentation"*, arXiv preprint:2410.01088v2, 2024.

[207] Xing Wu, Shangwen Lv, Liangjun Zang, Jizhong Han, Songlin Hu, *"Conditional BERT Contextual Augmentation"*, arXiv preprint:1812.06705v1, 2018.

[208] Lingling Xu, Haoran Xie, S. Joe Qin, Fu Lee Wang, Xiaohui Tao, *"Exploring ChatGPT-based Augmentation Strategies for Contrastive  Aspect-based Sentiment Analysis"*, arXiv preprint:2409.11218v1, 2024.

[209] Sosuke Kobayashi, *"Contextual Augmentation: Data Augmentation by Words with Paradigmatic  Relations"*, arXiv preprint:1805.06201v1, 2018.

[210] Ed S. Ma, *"Investigating Masking-based Data Generation in Language Models"*, arXiv preprint:2307.00008v1, 2023.

[211] Domagoj Pluščec, Jan Šnajder, *"Data Augmentation for Neural NLP"*, arXiv preprint:2302.11412v1, 2023.

[212] Ranjan Sapkota, Shaina Raza, Maged Shoman, Achyut Paudel, Manoj Karkee, *"Image, Text, and Speech Data Augmentation using Multimodal LLMs for Deep  Learning: A Survey"*, arXiv preprint:2501.18648v1, 2025.

[213] Yue Zhou, Chenlu Guo, Xu Wang, Yi Chang, Yuan Wu, *"A Survey on Data Augmentation in Large Model Era"*, arXiv preprint:2401.15422v2, 2024.

[214] Byeong-Cheol Jo, Tak-Sung Heo, Yeongjoon Park, Yongmin Yoo, Won Ik Cho, Kyungsun Kim, *"DAGAM: Data Augmentation with Generation And Modification"*, arXiv preprint:2204.02633v1, 2022.

[215] Tianqing Fang, Wenxuan Zhou, Fangyu Liu, Hongming Zhang, Yangqiu Song, Muhao Chen, *"On-the-fly Denoising for Data Augmentation in Natural Language  Understanding"*, arXiv preprint:2212.10558v2, 2022.

[216] Yichuan Li, Kaize Ding, Jianling Wang, Kyumin Lee, *"Empowering Large Language Models for Textual Data Augmentation"*, arXiv preprint:2404.17642v1, 2024.

[217] Seonghyeon Ye, Jiseon Kim, Alice Oh, *"Efficient Contrastive Learning via Novel Data Augmentation and  Curriculum Learning"*, arXiv preprint:2109.05941v2, 2021.

[218] Zhenhua Liu, Tong Zhu, Jianxiang Xiang, Wenliang Chen, *"Controllable and Diverse Data Augmentation with Large Language Model for  Low-Resource Open-Domain Dialogue Generation"*, arXiv preprint:2404.00361v1, 2024.

[219] Matthew Ciolino, David Noever, Josh Kalin, *"Back Translation Survey for Improving Text Augmentation"*, arXiv preprint:2102.09708v2, 2021.

[220] Baolin Peng, Chenguang Zhu, Michael Zeng, Jianfeng Gao, *"Data Augmentation for Spoken Language Understanding via Pretrained  Language Models"*, arXiv preprint:2004.13952v2, 2020.

[221] Yanru Qu, Dinghan Shen, Yelong Shen, Sandra Sajeev, Jiawei Han, Weizhu Chen, *"CoDA: Contrast-enhanced and Diversity-promoting Data Augmentation for  Natural Language Understanding"*, arXiv preprint:2010.08670v1, 2020.

[222] Heng Yang, Ke Li, *"BootAug: Boosting Text Augmentation via Hybrid Instance Filtering  Framework"*, arXiv preprint:2210.02941v2, 2022.

[223] Mingyang Yi, Lu Hou, Lifeng Shang, Xin Jiang, Qun Liu, Zhi-Ming Ma, *"Reweighting Augmented Samples by Minimizing the Maximal Expected Loss"*, arXiv preprint:2103.08933v1, 2021.

[224] Junfan Chen, Richong Zhang, Zheyan Luo, Chunming Hu, Yongyi Mao, *"Adversarial Word Dilution as Text Data Augmentation in Low-Resource  Regime"*, arXiv preprint:2305.09287v2, 2023.

[225] Sreyan Ghosh, Chandra Kiran Evuru, Sonal Kumar, S Ramaneswaran, S Sakshi, Utkarsh Tyagi, Dinesh Manocha, *"DALE: Generative Data Augmentation for Low-Resource Legal NLP"*, arXiv preprint:2310.15799v1, 2023.

[226] Minju Seo, Jinheon Baek, James Thorne, Sung Ju Hwang, *"Retrieval-Augmented Data Augmentation for Low-Resource Domain Tasks"*, arXiv preprint:2402.13482v1, 2024.

[227] Bosheng Ding, Chengwei Qin, Ruochen Zhao, Tianze Luo, Xinze Li, Guizhen Chen, Wenhan Xia, Junjie Hu, Anh Tuan Luu, Shafiq Joty, *"Data Augmentation using Large Language Models: Data Perspectives,  Learning Paradigms and Challenges"*, arXiv preprint:2403.02990v4, 2024.

[228] Daijun Ding, Li Dong, Zhichao Huang, Guangning Xu, Xu Huang, Bo Liu, Liwen Jing, Bowen Zhang, *"EDDA: A Encoder-Decoder Data Augmentation Framework for Zero-Shot Stance  Detection"*, arXiv preprint:2403.15715v1, 2024.

[229] Yova Kementchedjhieva, Adam Lopez, *"Indicatements that character language models learn English  morpho-syntactic units and regularities"*, arXiv preprint:1809.00066v1, 2018.

[230] Navid Rekabsaz, Nikolaos Pappas, James Henderson, Banriskhem K. Khonglah, Srikanth Madikeri, *"Regularization Advantages of Multilingual Neural Language Models for Low  Resource Domains"*, arXiv preprint:1906.01496v1, 2019.

[231] Jason Wei, Clara Meister, Ryan Cotterell, *"A Cognitive Regularizer for Language Modeling"*, arXiv preprint:2105.07144v3, 2021.

[232] Ta-Chung Chi, Ting-Han Fan, Alexander I. Rudnicky, Peter J. Ramadge, *"Transformer Working Memory Enables Regular Language Reasoning and  Natural Language Length Extrapolation"*, arXiv preprint:2305.03796v1, 2023.

[233] Tianlin Liu, Shangmin Guo, Leonardo Bianco, Daniele Calandriello, Quentin Berthet, Felipe Llinares, Jessica Hoffmann, Lucas Dixon, Michal Valko, Mathieu Blondel, *"Decoding-time Realignment of Language Models"*, arXiv preprint:2402.02992v2, 2024.

[234] Chun Feng, Joy Hsu, Weiyu Liu, Jiajun Wu, *"Naturally Supervised 3D Visual Grounding with Language-Regularized  Concept Learners"*, arXiv preprint:2404.19696v1, 2024.

[235] Clara Meister, Elizabeth Salesky, Ryan Cotterell, *"Generalized Entropy Regularization or: There's Nothing Special about  Label Smoothing"*, arXiv preprint:2005.00820v2, 2020.

[236] Tatsuya Hiraoka, *"MaxMatch-Dropout: Subword Regularization for WordPiece"*, arXiv preprint:2209.04126v1, 2022.

[237] Stephen Merity, Bryan McCann, Richard Socher, *"Revisiting Activation Regularization for Language RNNs"*, arXiv preprint:1708.01009v1, 2017.

[238] Shiwen Ni, Min Yang, Ruifeng Xu, Chengming Li, Xiping Hu, *"Layer-wise Regularized Dropout for Neural Language Models"*, arXiv preprint:2402.16361v1, 2024.

[239] Cheolhyoung Lee, Kyunghyun Cho, Wanmo Kang, *"Mixout: Effective Regularization to Finetune Large-scale Pretrained  Language Models"*, arXiv preprint:1909.11299v2, 2019.

[240] Zihan Liu, Genta Indra Winata, Peng Xu, Zhaojiang Lin, Pascale Fung, *"Cross-lingual Spoken Language Understanding with Regularized  Representation Alignment"*, arXiv preprint:2009.14510v1, 2020.

[241] Luca Malagutti, Andrius Buinovskij, Anej Svete, Clara Meister, Afra Amini, Ryan Cotterell, *"The Role of $n$-gram Smoothing in the Age of Neural Networks"*, arXiv preprint:2403.17240v2, 2024.

[242] Iñigo Parra, *"Morphological Typology in BPE Subword Productivity and Language Modeling"*, arXiv preprint:2410.23656v1, 2024.

[243] Vanessa Ferdinand, Simon Kirby, Kenny Smith, *"The cognitive roots of regularization in language"*, arXiv preprint:1703.03442v2, 2017.

[244] Rochelle Choenni, Dan Garrette, Ekaterina Shutova, *"Data-Efficient Cross-Lingual Transfer with Language-Specific Subnetworks"*, arXiv preprint:2211.00106v1, 2022.

[245] Masoud Monajatipoor, Liunian Harold Li, Mozhdeh Rouhsedaghat, Lin F. Yang, Kai-Wei Chang, *"MetaVL: Transferring In-Context Learning Ability From Language Models to  Vision-Language Models"*, arXiv preprint:2306.01311v1, 2023.

[246] Malte Ostendorff, Georg Rehm, *"Efficient Language Model Training through Cross-Lingual and Progressive  Transfer Learning"*, arXiv preprint:2301.09626v1, 2023.

[247] Tianze Hua, Tian Yun, Ellie Pavlick, *"mOthello: When Do Cross-Lingual Representation Alignment and  Cross-Lingual Transfer Emerge in Multilingual Models?"*, arXiv preprint:2404.12444v1, 2024.

[248] Kaushal Kumar Maurya, Maunendra Sankar Desarkar, *"Meta-X$_{NLG}$: A Meta-Learning Approach Based on Language Clustering  for Zero-Shot Cross-Lingual Transfer and Generation"*, arXiv preprint:2203.10250v1, 2022.

[249] Razan Baltaji, Saurabh Pujar, Louis Mandel, Martin Hirzel, Luca Buratti, Lav Varshney, *"Learning Transfers over Several Programming Languages"*, arXiv preprint:2310.16937v2, 2023.

[250] Shourav B. Rabbani, Ibna Kowsar, Manar D. Samad, *"Transfer Learning of Tabular Data by Finetuning Large Language Models"*, arXiv preprint:2501.06863v1, 2025.

[251] Evangelia Gogoulou, Ariel Ekgren, Tim Isbister, Magnus Sahlgren, *"Cross-lingual Transfer of Monolingual Models"*, arXiv preprint:2109.07348v2, 2021.

[252] Ling Ge, Chunming Hu, Guanghui Ma, Jihong Liu, Hong Zhang, *"DA-Net: A Disentangled and Adaptive Network for Multi-Source  Cross-Lingual Transfer Learning"*, arXiv preprint:2403.04158v1, 2024.

[253] Juuso Eronen, Michal Ptaszynski, Karol Nowakowski, Zheng Lin Chia, Fumito Masui, *"Improving Polish to English Neural Machine Translation with Transfer  Learning: Effects of Data Volume and Language Similarity"*, arXiv preprint:2306.00660v1, 2023.

[254] Tobias Strangmann, Lennart Purucker, Jörg K. H. Franke, Ivo Rapant, Fabio Ferreira, Frank Hutter, *"Transfer Learning for Finetuning Large Language Models"*, arXiv preprint:2411.01195v1, 2024.

[255] Michael Beukman, Manuel Fokam, *"Analysing Cross-Lingual Transfer in Low-Resourced African Named Entity  Recognition"*, arXiv preprint:2309.05311v1, 2023.

[256] Haneul Yoo, Cheonbok Park, Sangdoo Yun, Alice Oh, Hwaran Lee, *"Code-Switching Curriculum Learning for Multilingual Transfer in LLMs"*, arXiv preprint:2411.02460v1, 2024.

[257] Zuchao Li, Kevin Parnow, Hai Zhao, Zhuosheng Zhang, Rui Wang, Masao Utiyama, Eiichiro Sumita, *"Cross-lingual Transferring of Pre-trained Contextualized Language Models"*, arXiv preprint:2107.12627v1, 2021.

[258] Barret Zoph, Deniz Yuret, Jonathan May, Kevin Knight, *"Transfer Learning for Low-Resource Neural Machine Translation"*, arXiv preprint:1604.02201v1, 2016.

[259] Benjamin Muller, Deepanshu Gupta, Siddharth Patwardhan, Jean-Philippe Fauconnier, David Vandyke, Sachin Agarwal, *"Languages You Know Influence Those You Learn: Impact of Language  Characteristics on Multi-Lingual Text-to-Text Transfer"*, arXiv preprint:2212.01757v1, 2022.

[260] Evangelia Gogoulou, Timothée Lesort, Magnus Boman, Joakim Nivre, *"Continual Learning Under Language Shift"*, arXiv preprint:2311.01200v4, 2023.

