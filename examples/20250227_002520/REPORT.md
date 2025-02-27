# Designing an LLM-Based Research Assistant: A Detailed Scientific Report

This report synthesizes information from multiple research papers to provide a comprehensive guide on designing an LLM-based research assistant. The report covers key architectural components, design choices, ethical considerations, and evaluation methods, citing the corresponding papers for each argument.

## I. Core Functionalities and Design Principles

To effectively design an LLM-based research assistant, it is crucial to understand the core functionalities and ethical considerations.

### A. Defining Core Functionalities

The first step is to define the specific research tasks the assistant will perform. These tasks can be grouped into three key roles for LLMs in academic knowledge creation [1]:

1.  **GPT as a Co-Writer:**
    *   **Functions:** Generating drafts of research papers, presentations, scanning and evaluating current academic research, identifying and prioritizing research directions [1].
    *   **Design Implications:** The research assistant should be able to:
        *   Generate drafts or sections of research papers [1].
        *   Provide summaries of existing research papers [1].
        *   Suggest potential research directions based on current literature [1].
        *   Assist with outlining and structuring research papers [1].
        *   Assist with writing engaging captions for data visualizations [1].
2.  **GPT as a Research Assistant:**
    *   **Functions:** Supporting literature search, data preparation, transformation, and synthesis, determining the approach to create academic contribution, designing approach protocol, performing exploratory evaluation, documenting decisions and findings, identifying academic and industry implications of findings [1].
    *   **Design Implications:** The research assistant should be able to:
        *   Perform advanced literature searches and provide summaries of relevant articles [1].
        *   Assist with data cleaning, formatting, and exploratory analysis [1].
        *   Generate items for scale development in quantitative research [1].
        *   Help researchers explore research options and potential respondent behavior [1].
        *   Automate data preprocessing tasks [1].
        *   Design large language experiments [1].
3.  **GPT as a Respondent:**
    *   **Functions:** Acting as a source of simulated respondents and systems, providing perspectives on issues, answering interview questions, participating in experiments [1].
    *   **Design Implications:** The research assistant could:
        *   Provide simulated responses to research questions for exploratory purposes [1].
        *   Replicate findings from economic experiments [1].
        *   Act as a surrogate user in conversational search systems [1].
        *   Simulate responders for sensitive topics [1].
        *   Create simulated characters and simulate interactions among them [1].

### B. Prioritizing Ethical Considerations

Ethical considerations are crucial to ensure responsible adoption [7, 1]. Key design considerations include [1]:

*   **Transparency:** Clearly indicate when the LLM is providing information or suggestions [2]. The UI should make it clear when the LLM is generating content or making suggestions. Provide ways for researchers to inspect the sources of information and the reasoning behind the LLM's outputs [7]. Papers may include a structured description of prompts and responses along with a recording of text generation in real time to enable replication of research [1].
*   **Authorship Attribution:** Implement mechanisms for transparency and proper attribution when using AI-generated content [1]. Academic publishers require the human authors to take full responsibility for the manuscript’s content [1].
*   **Detection of Plagiarism and Inaccuracies:** The system should be able to detect and flag potential plagiarism or inaccuracies [1]. Implement plagiarism detection tools [7].
*   **Awareness of Biases:** Recognize that LLMs may exhibit biases based on their training data [1]. Your design should encourage diverse exploration and critical thinking [2].
*   **Limitations of Simulated Responses:** Provide clear disclaimers about the limitations of simulated responses and emphasize that they should not be generalized to humans without further validation [1]. Findings should be limited to the AI domain only [1].
*   **Data Privacy:** Comply with data protection laws and ethical guidelines, especially when handling sensitive information [8].
*   **Human Oversight:** Emphasize the importance of human review and validation of the LLM's outputs [2].

## II. Architectural Components and Implementation Details

### A. User Interface (UI) and Interaction

*   **Prompt-Based Interaction:** The UI should be based on natural language prompts [1]. Direct manipulation should be considered [7].
*   **Intuitive Design:** Focus on an intuitive and user-friendly interface, possibly with pop-up tips and clear guidance [5].
*   **Accessibility:** Ensure that the research assistant is accessible to users with varying levels of technological literacy and access to resources [8].

### B. Knowledge Acquisition and Management

1.  **Data Sources:**
    *   **Academic Databases:** Integrate with academic search engines and databases like Scopus, Google Scholar, Semantic Scholar, and PubMed [1].
    *   **Knowledge Graphs:** Utilize knowledge graphs like Wikidata, DBpedia, and domain-specific KGs to provide structured knowledge [3].
    *   **User-Provided Data:** Allow users to upload their own source documents in various formats (PDF, JSON, TXT) [5]. Use OCR (Optical Character Recognition) for scanned PDFs [5].
2.  **Data Processing:**
    *   **Text Extraction:** Employ libraries like PYPDF2, PDFMiner, or specialized OCR tools to extract text from PDFs [9].
    *   **Data Cleaning:** Implement data cleaning and preprocessing steps to ensure uniformity and remove noise, removing or replacing specific characters, addressing encoding issues, and standardizing formats (dates, numbers, etc.) [12].
    *   **Semantic Scholar API:** Used for retrieving research papers and extracting information [4].
    *   **Hugging Face's Datasets and Model Hub:** Used for retrieving datasets and pre-trained models [4].

### C. Question Generation

1.  **Prompt Engineering:**
    *   **Task descriptions:** Clearly define the task for the LLM [3].
    *   **Examples (few-shot learning):** Provide examples to guide the LLM [3].
    *   **Instructions:** Provide specific instructions to optimize performance [3].
    *   **Chain-of-Thought (CoT):** Employ CoT prompting strategies to guide the LLM's reasoning [3].

### D. Answer Generation

1.  **Retrieval-Augmented Generation (RAG):**
    *   **Integrating Knowledge:** Use RAG to connect the LLM to relevant databases, research papers, and other knowledge sources. Consider task-specific retrieval methods [6].
    *   **Vector Databases:** Use vector databases to store and retrieve relevant research papers quickly [3].
    *   **Prompt Engineering:** The quality of the LLM’s output is highly dependent on the prompt used. A well-designed prompt can guide the LLM to generate more relevant and coherent summaries [9].
2.  **Reflective Incremental Generator:**
     * This module is responsible for generating the comparative literature summary. The generator consists of two sub-modules:
         * Comparative Summarizer: iteratively generates summaries by considering the relationship between the current reference paper, the proposed work, and the existing partial literature review.
         * Reflective Evaluator: filters the generated summaries to ensure quality and stability [12].
3.  **LLM Selection and Configuration:**
    *   **Consider GPT-3.5, GPT-4, or open-source alternatives:** Open-source models allow more control but require more resources [3].
    *   **Joint Training:** This emphasizes the importance of training the different components of the system together, to ensure that they work well in concert [13].
    *   **Pre-training:** Pre-training certain modules can help to stabilize and improve the training process [13].

### E. Evaluation

1.  **Hallucination Mitigation:**
    *   **Factuality Check:** Use techniques to detect and mitigate adversarial inputs and hallucinations. Use external knowledge sources for verification [3]. Implement mechanisms to detect and flag potentially false or fabricated content [7].
    *   **Hallucination Detection:** Implement robust hallucination detection and mitigation strategies [10]. Use external knowledge sources for verification [3].
2.  **Evaluation Metrics:**
    *   **ROUGE Scores:** Used to quantitatively evaluate the performance of summarization approaches [9].
    *   **G-Score:** A multidimensional LLM-based metric that considers consistency, coherence, comparative analysis, integrity, fluency, and cite accuracy [12].
    *   **Metrics** like accuracy, completeness, and efficiency of the research assistant on a set of benchmark tasks [3].
    *   **User Evaluation:** User feedback to evaluate the system [9].

## III. Workflow and Interaction Design

### A. Workflow

The "MLR-Copilot" paper outlines a three-stage workflow [4]:

1.  **Research Idea Generation:**
    *   Analyze the literature to extract essential information, including research tasks `t`, research gaps `g`, and keywords `k = {k1, k2, ..., km}` using LLMs.
    *   Create prompt `P = {c, t, g, k}` to retrieve recent related works `R = {r1, r2, ..., rl}`.
    *   Use prompt `P1 = {P, R} -> h` to generate hypotheses `h` based on identified trends and gaps.
    *   Create a detailed experimental plan using prompt `P2 = {P1, h} -> e`.
    *   Define a research idea as `RI = {P, R, h, e}`.
2.  **Experiment Implementation:**
    *   Retrieve prototype implementation `I` from the original paper.
    *   Retrieve suitable models `M∇` from a model repository `M = {M1, M2, ..., Mp}`.
    *   Identify and retrieve relevant datasets `D ∈ {D1, D2, ..., Dq}`.
    *   Modify the code to ensure compatibility with selected models and datasets.
    *   Integrate the retrieved models, datasets, and prototype code into a cohesive experimental setup with experimental implementation `(I, M∇, D) -> S`.
3.  **Implementation Execution:**
    *   Execute the experimental setups `(I, M∇, D) -> S` under the management of ExperimentAgent.
    *   Oversee the allocation of computational resources.
    *   Monitor the progress and performance of the experiments.
    *   Integrate mechanisms for human feedback, allowing researchers to provide input and adjustments during the execution phase.
    *   Enable researchers to refine their hypotheses and experimental designs based on intermediate and final execution results.

### B. Modalities of Human-AI Interaction

*   **AI as the Primary Inventor:** The LLM provides all the input, and the human simply executes the instructions. This could democratize research by enabling non-specialists [2].
*   **Collaborative Exploration:** The LLM augments the human’s expertise by providing interdisciplinary knowledge and wide-ranging information. This is probably the most valuable for a research assistant [2].
*   **AI as a Funnel/Refiner:** The LLM helps to refine the design process and provides technical input, while the human remains the inventor or scientist [2].

## IV. Iterative Development and Refinement

*   **Feedback Loops:** The implementation and execution process are iterative [4].
*   **Human Feedback Integration:** Integrate mechanisms for human feedback, allowing researchers to provide input and adjustments during the execution phase [4].
*   **Metrics:** Use metrics like ROUGE scores (for summarization) or other task-specific metrics to evaluate the performance of your system [9].
*   **Continuous Evaluation and Improvement:** Continuously evaluate the performance of the research assistant using both automatic metrics and human feedback. Refine the prompts, models, and algorithms based on the evaluation results [12].

## V. Long-Term Considerations

*   **Skills for LLM-Based Research:** Skills to use LLMs will be part of researchers’ near future core competencies. Research methods modules at universities will need to incorporate LLM-based research methodologies and skills in order to equip the future researchers with the necessary research skills [1].
*   **Impact on Research Skills:** Monitor the long-term effects of LLM use on researchers’ skills (critical thinking, writing, data analysis) [7].
*   **Community Norms:** Engage with the research community to develop and refine guidelines for responsible LLM usage [7].

By following this comprehensive guide, you can design an LLM-based research assistant that effectively supports and augments human researchers in their work.

### References
[1] Nigel Williams, Stanislav Ivanov, Dimitrios Buhalis, *"Algorithmic Ghost in the Research Shell: Large Language Models and  Academic Knowledge Creation in Management Research"*, arXiv preprint:2303.07304v1, 2023.

[2] Francesco Stella, Cosimo Della Santina, Josie Hughes, *"Can Large Language Models design a Robot?"*, arXiv preprint:2303.15324v1, 2023.

[3] Katikapalli Subramanyam Kalyan, *"A Survey of GPT-3 Family Large Language Models Including ChatGPT and  GPT-4"*, arXiv preprint:2310.12321v1, 2023.

[4] Ruochen Li, Teerth Patel, Qingyun Wang, Xinya Du, *"MLR-Copilot: Autonomous Machine Learning Research based on Large  Language Models Agents"*, arXiv preprint:2408.14033v2, 2024.

[5] Thomas Übellacker, *"AcademiaOS: Automating Grounded Theory Development in Qualitative  Research with Large Language Models"*, arXiv preprint:2403.08844v1, 2024.

[6] Hashmath Shaik, Alex Doboli, *"An Overview and Discussion on Using Large Language Models for  Implementation Generation of Solutions to Open-Ended Problems"*, arXiv preprint:2501.00562v2, 2024.

[7] Zhehui Liao, Maria Antoniak, Inyoung Cheong, Evie Yu-Yen Cheng, Ai-Heng Lee, Kyle Lo, Joseph Chee Chang, Amy X. Zhang, *"LLMs as Research Tools: A Large Scale Survey of Researchers' Usage and  Perceptions"*, arXiv preprint:2411.05025v1, 2024.

[8] Jiangfeng Liu, Ziyi Wang, Jing Xie, Lei Pei, *"From ChatGPT, DALL-E 3 to Sora: How has Generative AI Changed Digital  Humanities Research and Services?"*, arXiv preprint:2404.18518v1, 2024.

[9] Nurshat Fateh Ali, Md. Mahdi Mohtasim, Shakil Mosharrof, T. Gopi Krishna, *"Automated Literature Review Using NLP Techniques and LLM-Based  Retrieval-Augmented Generation"*, arXiv preprint:2411.18583v1, 2024.

[10] Xuemei Tang, Xufeng Duan, Zhenguang G. Cai, *"Are LLMs Good Literature Review Writers? Evaluating the Literature  Review Writing Ability of Large Language Models"*, arXiv preprint:2412.13612v2, 2024.

[11] Rock Yuren Pang, Hope Schroeder, Kynnedy Simone Smith, Solon Barocas, Ziang Xiao, Emily Tseng, Danielle Bragg, *"Understanding the LLM-ification of CHI: Unpacking the Impact of LLMs at  CHI through a Systematic Literature Review"*, arXiv preprint:2501.12557v1, 2025.

[12] Yutong Li, Lu Chen, Aiwei Liu, Kai Yu, Lijie Wen, *"ChatCite: LLM Agent with Human Workflow Guidance for Comparative  Literature Summary"*, arXiv preprint:2403.02574v1, 2024.

[13] Yuntong Hu, Zhuofeng Li, Zheng Zhang, Chen Ling, Raasikh Kanjiani, Boxin Zhao, Liang Zhao, *"HiReview: Hierarchical Taxonomy-Driven Automatic Literature Review  Generation"*, arXiv preprint:2410.03761v1, 2024.

[14] David A. Tovar, *"AI Literature Review Suite"*, arXiv preprint:2308.02443v1, 2023.

[15] Shubham Agarwal, Gaurav Sahu, Abhay Puri, Issam H. Laradji, Krishnamurthy DJ Dvijotham, Jason Stanley, Laurent Charlin, Christopher Pal, *"LLMs for Literature Review: Are we there yet?"*, arXiv preprint:2412.15249v1, 2024.

[16] Teo Susnjak, *"PRISMA-DFLLM: An Extension of PRISMA for Systematic Literature Reviews  using Domain-specific Finetuned Large Language Models"*, arXiv preprint:2306.14905v1, 2023.

[17] Chao-Chun Hsu, Erin Bransom, Jenna Sparks, Bailey Kuehl, Chenhao Tan, David Wadden, Lucy Lu Wang, Aakanksha Naik, *"CHIME: LLM-Assisted Hierarchical Organization of Scientific Studies for  Literature Review Support"*, arXiv preprint:2407.16148v1, 2024.

[18] Moritz Staudinger, Wojciech Kusa, Florina Piroi, Aldo Lipani, Allan Hanbury, *"A Reproducibility and Generalizability Study of Large Language Models  for Query Generation"*, arXiv preprint:2411.14914v1, 2024.

[19] M. Namvarpour, A. Razi, *"Apprentices to Research Assistants: Advancing Research with Large  Language Models"*, arXiv preprint:2404.06404v1, 2024.

[20] Hye Sun Yun, Iain J. Marshall, Thomas A. Trikalinos, Byron C. Wallace, *"Appraising the Potential Uses and Harms of LLMs for Medical Systematic  Reviews"*, arXiv preprint:2305.11828v3, 2023.

[21] Dmitry Scherbakov, Nina Hubig, Vinita Jansari, Alexander Bakumenko, Leslie A. Lenert, *"The emergence of Large Language Models (LLM) as a tool in literature  reviews: an LLM automated systematic review"*, arXiv preprint:2409.04600v1, 2024.

[22] Teo Susnjak, Peter Hwang, Napoleon H. Reyes, Andre L. C. Barczak, Timothy R. McIntosh, Surangika Ranathunga, *"Automating Research Synthesis with Domain-Specific Large Language Model  Fine-Tuning"*, arXiv preprint:2404.08680v1, 2024.

[23] Zhi Zhang, Yan Liu, Sheng-hua Zhong, Gong Chen, Yu Yang, Jiannong Cao, *"Mixture of Knowledge Minigraph Agents for Literature Review Generation"*, arXiv preprint:2411.06159v3, 2024.

[24] Yixuan Weng, Minjun Zhu, Guangsheng Bao, Hongbo Zhang, Jindong Wang, Yue Zhang, Linyi Yang, *"CycleResearcher: Improving Automated Research via Automated Review"*, arXiv preprint:2411.00816v1, 2024.

[25] Shubham Agarwal, Issam H. Laradji, Laurent Charlin, Christopher Pal, *"LitLLM: A Toolkit for Scientific Literature Review"*, arXiv preprint:2402.01788v1, 2024.

[26] Joaquin Ramirez-Medina, Mohammadmehdi Ataei, Alidad Amirfazli, *"Accelerating Scientific Research Through a Multi-LLM Framework"*, arXiv preprint:2502.07960v1, 2025.

[27] Boming Xia, Qinghua Lu, Liming Zhu, Zhenchang Xing, Dehai Zhao, Hao Zhang, *"An Evaluation-Driven Approach to Designing LLM Agents: Process and  Architecture"*, arXiv preprint:2411.13768v1, 2024.

[28] Hongye An, Arpit Narechania, Emily Wall, Kai Xu, *"vitaLITy 2: Reviewing Academic Literature Using Large Language Models"*, arXiv preprint:2408.13450v1, 2024.

[29] Yu Zhang, Xiusi Chen, Bowen Jin, Sheng Wang, Shuiwang Ji, Wei Wang, Jiawei Han, *"A Comprehensive Survey of Scientific Large Language Models and Their  Applications in Scientific Discovery"*, arXiv preprint:2406.10833v3, 2024.

[30] Ziming Luo, Zonglin Yang, Zexin Xu, Wei Yang, Xinya Du, *"LLM4SR: A Survey on Large Language Models for Scientific Research"*, arXiv preprint:2501.04306v1, 2025.

[31] Orcun Yildiz, Tom Peterka, *"Do Large Language Models Speak Scientific Workflows?"*, arXiv preprint:2412.10606v2, 2024.

[32] Jakub Lála, Odhran O'Donoghue, Aleksandar Shtedritski, Sam Cox, Samuel G. Rodriques, Andrew D. White, *"PaperQA: Retrieval-Augmented Generative Agent for Scientific Research"*, arXiv preprint:2312.07559v2, 2023.

[33] Jinheon Baek, Sujay Kumar Jauhar, Silviu Cucerzan, Sung Ju Hwang, *"ResearchAgent: Iterative Research Idea Generation over Scientific  Literature with Large Language Models"*, arXiv preprint:2404.07738v2, 2024.

[34] Hamed Babaei Giglou, Jennifer D'Souza, Sören Auer, *"LLMs4Synthesis: Leveraging Large Language Models for Scientific  Synthesis"*, arXiv preprint:2409.18812v1, 2024.

[35] Yubo Ma, Zhibin Gou, Junheng Hao, Ruochen Xu, Shuohang Wang, Liangming Pan, Yujiu Yang, Yixin Cao, Aixin Sun, Hany Awadalla, Weizhu Chen, *"SciAgent: Tool-augmented Language Models for Scientific Reasoning"*, arXiv preprint:2402.11451v2, 2024.

[36] Huy Quoc To, Ming Liu, Guangyan Huang, *"Towards Efficient Large Language Models for Scientific Text: A Review"*, arXiv preprint:2408.10729v1, 2024.

[37] Tal Ifargan, Lukas Hafner, Maor Kern, Ori Alcalay, Roy Kishony, *"Autonomous LLM-driven research from data to human-verifiable research  papers"*, arXiv preprint:2404.17605v1, 2024.

[38] Dayong Wu, Jiaqi Li, Baoxin Wang, Honghong Zhao, Siyuan Xue, Yanjie Yang, Zhijun Chang, Rui Zhang, Li Qian, Bo Wang, Shijin Wang, Zhixiong Zhang, Guoping Hu, *"SparkRA: A Retrieval-Augmented Knowledge Service System Based on Spark  Large Language Model"*, arXiv preprint:2408.06574v1, 2024.

[39] Oskar Wysocki, Magdalena Wysocka, Danilo Carvalho, Alex Teodor Bogatu, Danilo Miranda Gusicuma, Maxime Delmas, Harriet Unsworth, Andre Freitas, *"An LLM-based Knowledge Synthesis and Scientific Reasoning Framework for  Biomedical Discovery"*, arXiv preprint:2406.18626v1, 2024.

[40] Marissa Radensky, Simra Shahid, Raymond Fok, Pao Siangliulue, Tom Hope, Daniel S. Weld, *"Scideator: Human-LLM Scientific Idea Generation Grounded in  Research-Paper Facet Recombination"*, arXiv preprint:2409.14634v3, 2024.

[41] Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Zicheng Liu, Emad Barsoum, *"Agent Laboratory: Using LLM Agents as Research Assistants"*, arXiv preprint:2501.04227v1, 2025.

[42] James Boyko, Joseph Cohen, Nathan Fox, Maria Han Veiga, Jennifer I-Hsiu Li, Jing Liu, Bernardo Modenesi, Andreas H. Rauch, Kenneth N. Reid, Soumi Tribedi, Anastasia Visheratina, Xin Xie, *"An Interdisciplinary Outlook on Large Language Models for Scientific  Research"*, arXiv preprint:2311.04929v1, 2023.

[43] Tianyang Gu, Jingjin Wang, Zhihao Zhang, HaoHong Li, *"LLMs can Realize Combinatorial Creativity: Generating Creative Ideas via  LLMs for Scientific Research"*, arXiv preprint:2412.14141v2, 2024.

[44] Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Dongsheng Li, Deqing Yang, *"EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary  Algorithms"*, arXiv preprint:2406.14228v2, 2024.

[45] Haolin Jin, Linghan Huang, Haipeng Cai, Jun Yan, Bo Li, Huaming Chen, *"From LLMs to LLM-based Agents for Software Engineering: A Survey of  Current, Challenges and Future"*, arXiv preprint:2408.02479v1, 2024.

[46] Kai Mei, Xi Zhu, Wujiang Xu, Wenyue Hua, Mingyu Jin, Zelong Li, Shuyuan Xu, Ruosong Ye, Yingqiang Ge, Yongfeng Zhang, *"AIOS: LLM Agent Operating System"*, arXiv preprint:2403.16971v3, 2024.

[47] Zhiwei Liu, Weiran Yao, Jianguo Zhang, Liangwei Yang, Zuxin Liu, Juntao Tan, Prafulla K. Choubey, Tian Lan, Jason Wu, Huan Wang, Shelby Heinecke, Caiming Xiong, Silvio Savarese, *"AgentLite: A Lightweight Library for Building and Advancing  Task-Oriented LLM Agent System"*, arXiv preprint:2402.15538v1, 2024.

[48] Peiyuan Feng, Yichen He, Guanhua Huang, Yuan Lin, Hanchong Zhang, Yuchen Zhang, Hang Li, *"AGILE: A Novel Reinforcement Learning Framework of LLM Agents"*, arXiv preprint:2405.14751v2, 2024.

[49] Kuan Wang, Yadong Lu, Michael Santacroce, Yeyun Gong, Chao Zhang, Yelong Shen, *"Adapting LLM Agents with Universal Feedback in Communication"*, arXiv preprint:2310.01444v3, 2023.

[50] Yu Shang, Yu Li, Keyu Zhao, Likai Ma, Jiahe Liu, Fengli Xu, Yong Li, *"AgentSquare: Automatic LLM Agent Search in Modular Design Space"*, arXiv preprint:2410.06153v2, 2024.

[51] Maxwell Crouse, Ibrahim Abdelaziz, Ramon Astudillo, Kinjal Basu, Soham Dan, Sadhana Kumaravel, Achille Fokoue, Pavan Kapanipathi, Salim Roukos, Luis Lastras, *"Formally Specifying the High-Level Behavior of LLM-Based Agents"*, arXiv preprint:2310.08535v3, 2023.

[52] Wangchunshu Zhou, Yuchen Eleanor Jiang, Long Li, Jialong Wu, Tiannan Wang, Shi Qiu, Jintian Zhang, Jing Chen, Ruipu Wu, Shuai Wang, Shiding Zhu, Jiyu Chen, Wentao Zhang, Xiangru Tang, Ningyu Zhang, Huajun Chen, Peng Cui, Mrinmaya Sachan, *"Agents: An Open-source Framework for Autonomous Language Agents"*, arXiv preprint:2309.07870v3, 2023.

[53] Jeremy Harper, *"AutoGenesisAgent: Self-Generating Multi-Agent Systems for Complex Tasks"*, arXiv preprint:2404.17017v1, 2024.

[54] Ben Krause, Lucia Chen, Emmanuel Kahembwe, *"AutoGRAMS: Autonomous Graphical Agent Modeling Software"*, arXiv preprint:2407.10049v1, 2024.

[55] Abhishek Dutta, Yen-Che Hsiao, *"Towards Autonomous Agents: Adaptive-planning, Reasoning, and Acting in  Language Models"*, arXiv preprint:2408.06458v2, 2024.

[56] Weiran Yao, Shelby Heinecke, Juan Carlos Niebles, Zhiwei Liu, Yihao Feng, Le Xue, Rithesh Murthy, Zeyuan Chen, Jianguo Zhang, Devansh Arpit, Ran Xu, Phil Mui, Huan Wang, Caiming Xiong, Silvio Savarese, *"Retroformer: Retrospective Large Language Agents with Policy Gradient  Optimization"*, arXiv preprint:2308.02151v3, 2023.

[57] Sumedh Rasal, *"LLM Harmony: Multi-Agent Communication for Problem Solving"*, arXiv preprint:2401.01312v1, 2024.

[58] Tianbao Xie, Fan Zhou, Zhoujun Cheng, Peng Shi, Luoxuan Weng, Yitao Liu, Toh Jing Hua, Junning Zhao, Qian Liu, Che Liu, Leo Z. Liu, Yiheng Xu, Hongjin Su, Dongchan Shin, Caiming Xiong, Tao Yu, *"OpenAgents: An Open Platform for Language Agents in the Wild"*, arXiv preprint:2310.10634v1, 2023.

[59] Georg Wölflein, Dyke Ferber, Daniel Truhn, Ognjen Arandjelović, Jakob Nikolas Kather, *"LLM Agents Making Agent Tools"*, arXiv preprint:2502.11705v1, 2025.

[60] Haishuo Fang, Xiaodan Zhu, Iryna Gurevych, *"DARA: Decomposition-Alignment-Reasoning Autonomous Language Agent for  Question Answering over Knowledge Graphs"*, arXiv preprint:2406.07080v1, 2024.

[61] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen, *"A Survey on Large Language Model based Autonomous Agents"*, arXiv preprint:2308.11432v6, 2023.

[62] Faria Huq, Zora Zhiruo Wang, Frank F. Xu, Tianyue Ou, Shuyan Zhou, Jeffrey P. Bigham, Graham Neubig, *"CowPilot: A Framework for Autonomous and Human-Agent Collaborative Web  Navigation"*, arXiv preprint:2501.16609v2, 2025.

[63] Junda He, Christoph Treude, David Lo, *"LLM-Based Multi-Agent Systems for Software Engineering: Literature  Review, Vision and the Road Ahead"*, arXiv preprint:2404.04834v3, 2024.

[64] Thorsten Händler, *"Balancing Autonomy and Alignment: A Multi-Dimensional Taxonomy for  Autonomous LLM-powered Multi-Agent Architectures"*, arXiv preprint:2310.03659v1, 2023.

[65] Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, Yang Song, Chen Zhu, Hengshu Zhu, Ji-Rong Wen, *"KG-Agent: An Efficient Autonomous Agent Framework for Complex Reasoning  over Knowledge Graph"*, arXiv preprint:2402.11163v1, 2024.

[66] Feng Xiong, Xinguo Yu, Hon Wai Leong, *"AI-Empowered Human Research Integrating Brain Science and Social  Sciences Insights"*, arXiv preprint:2411.12761v1, 2024.

[67] Runlong Ye, Matthew Varona, Oliver Huang, Patrick Yung Kang Lee, Michael Liut, Carolina Nobre, *"The Design Space of Recent AI-assisted Research Tools for Ideation,  Sensemaking, and Scientific Creativity"*, arXiv preprint:2502.16291v1, 2025.

[68] César França, *"AI empowering research: 10 ways how science can benefit from AI"*, arXiv preprint:2307.10265v1, 2023.

[69] Khanh Nghiem, Anh Minh Nguyen, Nghi D. Q. Bui, *"Envisioning the Next-Generation AI Coding Assistants: Insights &  Proposals"*, arXiv preprint:2403.14592v1, 2024.

[70] Christoph Treude, Marco A. Gerosa, *"How Developers Interact with AI: A Taxonomy of Human-AI Collaboration in  Software Engineering"*, arXiv preprint:2501.08774v2, 2025.

[71] Angela Mastrianni, Hope Twede, Aleksandra Sarcevic, Jeremiah Wander, Christina Austin-Tse, Scott Saponas, Heidi Rehm, Ashley Mae Conard, Amanda K. Hall, *"AI-Enhanced Sensemaking: Exploring the Design of a Generative AI-Based  Assistant to Support Genetic Professionals"*, arXiv preprint:2412.15444v1, 2024.

[72] Ebtesam Al Haque, Chris Brown, Thomas D. LaToza, Brittany Johnson, *"Towards Decoding Developer Cognition in the Age of AI Assistants"*, arXiv preprint:2501.02684v1, 2025.

[73] Mike Perkins, Jasper Roe, *"Generative AI Tools in Academic Research: Applications and Implications  for Qualitative and Quantitative Research Methodologies"*, arXiv preprint:2408.06872v1, 2024.

[74] Alex Kontorovich, *"Notes on a Path to AI Assistance in Mathematical Reasoning"*, arXiv preprint:2310.02896v1, 2023.

[75] Marcin P. Joachimiak, Mark A. Miller, J. Harry Caufield, Ryan Ly, Nomi L. Harris, Andrew Tritt, Christopher J. Mungall, Kristofer E. Bouchard, *"The Artificial Intelligence Ontology: LLM-assisted construction of AI  concept hierarchies"*, arXiv preprint:2404.03044v1, 2024.

[76] Mahsa Shamsabadi, Jennifer D'Souza, *"A FAIR and Free Prompt-based Research Assistant"*, arXiv preprint:2405.14601v1, 2024.

[77] Jonan Richards, Mairieli Wessel, *"Bridging HCI and AI Research for the Evaluation of Conversational SE  Assistants"*, arXiv preprint:2502.07956v1, 2025.

[78] Agnia Sergeyuk, Yaroslav Golubev, Timofey Bryksin, Iftekhar Ahmed, *"Using AI-Based Coding Assistants in Practice: State of Affairs,  Perceptions, and Ways Forward"*, arXiv preprint:2406.07765v2, 2024.

[79] Zhicheng Lin, *"Techniques for supercharging academic writing with generative AI"*, arXiv preprint:2310.17143v3, 2023.

[80] Yared W. Bekele, *"GeoSim.AI: AI assistants for numerical simulations in geomechanics"*, arXiv preprint:2501.14186v1, 2025.

[81] Joseph Tu, Hilda Hadan, Derrick M. Wang, Sabrina A Sgandurra, Reza Hadi Mogavi, Lennart E. Nacke, *"Augmenting the Author: Exploring the Potential of AI Collaboration in  Academic Writing"*, arXiv preprint:2404.16071v1, 2024.

[82] Ryo Suzuki, Mar Gonzalez-Franco, Misha Sra, David Lindlbauer, *"Everyday AR through AI-in-the-Loop"*, arXiv preprint:2412.12681v1, 2024.

[83] Elisabeth Kirsten, Annalina Buckmann, Leona Lassak, Nele Borgert, Abraham Mhaidli, Steffen Becker, *"From Assistance to Autonomy -- A Researcher Study on the Potential of AI  Support for Qualitative Data Analysis"*, arXiv preprint:2501.19275v1, 2025.

[84] Victor Morel, Leonardo Iwaya, Simone Fischer-Hübner, *"SoK: A Classification for AI-driven Personalized Privacy Assistants"*, arXiv preprint:2502.07693v2, 2025.

[85] Meredith Dedema, Rongqian Ma, *"The collective use and perceptions of generative AI tools in digital  humanities research: Survey-based results"*, arXiv preprint:2404.12458v2, 2024.

[86] Mark Glickman, Yi Zhang, *"AI and Generative AI for Research Discovery and Summarization"*, arXiv preprint:2401.06795v2, 2024.

[87] Mohammad Amin Samadi, Spencer JaQuay, Jing Gu, Nia Nixon, *"The AI Collaborator: Bridging Human-AI Interaction in Educational and  Professional Settings"*, arXiv preprint:2405.10460v1, 2024.

[88] Haotian Li, Yun Wang, Huamin Qu, *"Where Are We So Far? Understanding Data Storytelling Tools from the  Perspective of Human-AI Collaboration"*, arXiv preprint:2309.15723v2, 2023.

[89] Zhuoyi Cheng, Pei Chen, Wenzheng Song, Hongbo Zhang, Zhuoshu Li, Lingyun Sun, *"An Exploratory Study on How AI Awareness Impacts Human-AI Design  Collaboration"*, arXiv preprint:2502.16833v1, 2025.

[90] Debayan Banerjee, Seid Muhie Yimam, Sushil Awale, Chris Biemann, *"ARDIAS: AI-Enhanced Research Management, Discovery, and Advisory System"*, arXiv preprint:2301.10577v1, 2023.

[91] Aung Pyae, *"The Human-AI Handshake Framework: A Bidirectional Approach to Human-AI  Collaboration"*, arXiv preprint:2502.01493v1, 2025.

[92] Ruoxi Xu, Yingfei Sun, Mengjie Ren, Shiguang Guo, Ruotong Pan, Hongyu Lin, Le Sun, Xianpei Han, *"AI for social science and social science of AI: A Survey"*, arXiv preprint:2401.11839v1, 2024.

[93] Wei Xu, Zaifeng Gao, *"Applying HCAI in developing effective human-AI teaming: A perspective  from human-AI joint cognitive systems"*, arXiv preprint:2307.03913v5, 2023.

[94] Abidullah Khan, Atefeh Shokrizadeh, Jinghui Cheng, *"Beyond Automation: How UI/UX Designers Perceive AI as a Creative Partner  in the Divergent Thinking Stages"*, arXiv preprint:2501.18778v1, 2025.

[95] Zifan Wang, Kotaro Funakoshi, Manabu Okumura, *"Automatic Answerability Evaluation for Question Generation"*, arXiv preprint:2309.12546v2, 2023.

[96] Fréderic Godin, Anjishnu Kumar, Arpit Mittal, *"Learning When Not to Answer: A Ternary Reward Structure for  Reinforcement Learning based Question Answering"*, arXiv preprint:1902.10236v2, 2019.

[97] Weiping Fu, Bifan Wei, Jianxiang Hu, Zhongmin Cai, Jun Liu, *"QGEval: Benchmarking Multi-dimensional Evaluation for Question  Generation"*, arXiv preprint:2406.05707v2, 2024.

[98] Fangyuan Xu, Yixiao Song, Mohit Iyyer, Eunsol Choi, *"A Critical Evaluation of Evaluations for Long-form Question Answering"*, arXiv preprint:2305.18201v1, 2023.

[99] Iulian Vlad Serban, Alberto García-Durán, Caglar Gulcehre, Sungjin Ahn, Sarath Chandar, Aaron Courville, Yoshua Bengio, *"Generating Factoid Questions With Recurrent Neural Networks: The 30M  Factoid Question-Answer Corpus"*, arXiv preprint:1603.06807v2, 2016.

[100] Hwanhee Lee, Seunghyun Yoon, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Joongbo Shin, Kyomin Jung, *"KPQA: A Metric for Generative Question Answering Using Keyphrase Weights"*, arXiv preprint:2005.00192v3, 2020.

[101] Matan Eyal, Tal Baumel, Michael Elhadad, *"Question Answering as an Automatic Evaluation Metric for News Article  Summarization"*, arXiv preprint:1906.00318v1, 2019.

[102] An Yang, Kai Liu, Jing Liu, Yajuan Lyu, Sujian Li, *"Adaptations of ROUGE and BLEU to Better Evaluate Machine Reading  Comprehension Task"*, arXiv preprint:1806.03578v1, 2018.

[103] Feng Liu, Tao Xiang, Timothy M. Hospedales, Wankou Yang, Changyin Sun, *"iVQA: Inverse Visual Question Answering"*, arXiv preprint:1710.03370v2, 2017.

[104] Michael Boratko, Xiang Lorraine Li, Rajarshi Das, Tim O'Gorman, Dan Le, Andrew McCallum, *"ProtoQA: A Question Answering Dataset for Prototypical Common-Sense  Reasoning"*, arXiv preprint:2005.00771v3, 2020.

[105] Ruosen Li, Ruochen Li, Barry Wang, Xinya Du, *"IQA-EVAL: Automatic Evaluation of Human-Model Interactive Question  Answering"*, arXiv preprint:2408.13545v2, 2024.

[106] Abdelghny Orogat, Isabelle Liu, Ahmed El-Roby, *"CBench: Towards Better Evaluation of Question Answering Over Knowledge  Graphs"*, arXiv preprint:2105.00811v1, 2021.

[107] Kaige Xie, Philippe Laban, Prafulla Kumar Choubey, Caiming Xiong, Chien-Sheng Wu, *"Do RAG Systems Cover What Matters? Evaluating and Optimizing Responses  with Sub-Question Coverage"*, arXiv preprint:2410.15531v1, 2024.

[108] Yun Joon Soh, Jishen Zhao, *"A Step Towards Mixture of Grader: Statistical Analysis of Existing  Automatic Evaluation Metrics"*, arXiv preprint:2410.10030v1, 2024.

[109] Naghmeh Farzi, Laura Dietz, *"An Exam-based Evaluation Approach Beyond Traditional Relevance Judgments"*, arXiv preprint:2402.00309v1, 2024.

[110] Preksha Nema, Mitesh M. Khapra, *"Towards a Better Metric for Evaluating Question Generation Systems"*, arXiv preprint:1808.10192v2, 2018.

[111] Khalil Mrini, Harpreet Singh, Franck Dernoncourt, Seunghyun Yoon, Trung Bui, Walter Chang, Emilia Farcas, Ndapa Nakashole, *"Medical Question Understanding and Answering with Knowledge Grounding  and Semantic Self-Supervision"*, arXiv preprint:2209.15301v1, 2022.

[112] Lingbo Mo, Besnik Fetahu, Oleg Rokhlenko, Shervin Malmasi, *"Controllable Decontextualization of Yes/No Question and Answers into  Factual Statements"*, arXiv preprint:2401.09775v1, 2024.

[113] Akchay Srivastava, Atif Memon, *"Towards Robust Evaluation: A Comprehensive Taxonomy of Datasets and  Metrics for Open Domain Question Answering in the Era of Large Language  Models"*, arXiv preprint:2406.13232v1, 2024.

[114] Reinald Kim Amplayo, Kellie Webster, Michael Collins, Dipanjan Das, Shashi Narayan, *"Query Refinement Prompts for Closed-Book Long-Form Question Answering"*, arXiv preprint:2210.17525v1, 2022.

[115] Sihui Yang, Keping Bi, Wanqing Cui, Jiafeng Guo, Xueqi Cheng, *"LINKAGE: Listwise Ranking among Varied-Quality References for  Non-Factoid QA Evaluation via LLMs"*, arXiv preprint:2409.14744v2, 2024.

[116] Thomas Scialom, Jacopo Staiano, *"Ask to Learn: A Study on Curiosity-driven Question Generation"*, arXiv preprint:1911.03350v1, 2019.

[117] Nigel Fernandez, Alexander Scarlatos, Andrew Lan, *"SyllabusQA: A Course Logistics Question Answering Dataset"*, arXiv preprint:2403.14666v2, 2024.

[118] Talha Chafekar, Aafiya Hussain, Grishma Sharma, Deepak Sharma, *"Exploring Answer Information Methods for Question Generation with  Transformers"*, arXiv preprint:2312.03483v1, 2023.

[119] Matteo Gabburo, Siddhant Garg, Rik Koncel Kedziorski, Alessandro Moschitti, *"SQUARE: Automatic Question Answering Evaluation using Multiple Positive  and Negative References"*, arXiv preprint:2309.12250v1, 2023.

[120] Shiyue Zhang, Mohit Bansal, *"Addressing Semantic Drift in Question Generation for Semi-Supervised  Question Answering"*, arXiv preprint:1909.06356v1, 2019.

[121] Chongyan Chen, Mengchen Liu, Noel Codella, Yunsheng Li, Lu Yuan, Danna Gurari, *"Fully Authentic Visual Question Answering Dataset from Online  Communities"*, arXiv preprint:2311.15562v4, 2023.

[122] Rongwu Xu, Xuan Qi, Zehan Qi, Wei Xu, Zhijiang Guo, *"DebateQA: Evaluating Question Answering on Debatable Knowledge"*, arXiv preprint:2408.01419v1, 2024.

[123] Mona Gandhi, Mustafa Omer Gul, Eva Prakash, Madeleine Grunde-McLaughlin, Ranjay Krishna, Maneesh Agrawala, *"Measuring Compositional Consistency for Video Question Answering"*, arXiv preprint:2204.07190v2, 2022.

[124] Ivo Lodovico Molina, Valdemar Švábenský, Tsubasa Minematsu, Li Chen, Fumiya Okubo, Atsushi Shimada, *"Comparison of Large Language Models for Generating Contextually Relevant  Questions"*, arXiv preprint:2407.20578v2, 2024.

[125] Juan Sequeda, Dean Allemang, Bryon Jacob, *"A Benchmark to Understand the Role of Knowledge Graphs on Large Language  Model's Accuracy for Question Answering on Enterprise SQL Databases"*, arXiv preprint:2311.07509v1, 2023.

[126] Man Luo, Shailaja Keyur Sampat, Riley Tallman, Yankai Zeng, Manuha Vancha, Akarshan Sajja, Chitta Baral, *"'Just because you are right, doesn't mean I am wrong': Overcoming a  Bottleneck in the Development and Evaluation of Open-Ended Visual Question  Answering (VQA) Tasks"*, arXiv preprint:2103.15022v2, 2021.

[127] Clemencia Siro, Yifei Yuan, Mohammad Aliannejadi, Maarten de Rijke, *"AGENT-CQ: Automatic Generation and Evaluation of Clarifying Questions  for Conversational Search with LLMs"*, arXiv preprint:2410.19692v1, 2024.

[128] Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, Ming-Wei Chang, *"ASQA: Factoid Questions Meet Long-Form Answers"*, arXiv preprint:2204.06092v2, 2022.

[129] Shailza Jolly, Sandro Pezzelle, Tassilo Klein, Andreas Dengel, Moin Nabi, *"The Wisdom of MaSSeS: Majority, Subjectivity, and Semantic Similarity in  the Evaluation of VQA"*, arXiv preprint:1809.04344v1, 2018.

[130] Shen Gao, Xiuying Chen, Zhaochun Ren, Dongyan Zhao, Rui Yan, *"Meaningful Answer Generation of E-Commerce Question-Answering"*, arXiv preprint:2011.07307v1, 2020.

[131] Royal Sequiera, Gaurav Baruah, Zhucheng Tu, Salman Mohammed, Jinfeng Rao, Haotian Zhang, Jimmy Lin, *"Exploring the Effectiveness of Convolutional Neural Networks for Answer  Selection in End-to-End Question Answering"*, arXiv preprint:1707.07804v1, 2017.

[132] Oscar Mañas, Benno Krojer, Aishwarya Agrawal, *"Improving Automatic VQA Evaluation Using Large Language Models"*, arXiv preprint:2310.02567v2, 2023.

[133] Ankit Shah, Srishti Singh, Shih-Yen Tao, *"Feature extraction and evaluation for BioMedical Question Answering"*, arXiv preprint:2105.14013v1, 2021.

[134] Martin Boyanov, Ivan Koychev, Preslav Nakov, Alessandro Moschitti, Giovanni Da San Martino, *"Building Chatbots from Forum Data: Model Selection Using Question  Answering Metrics"*, arXiv preprint:1710.00689v1, 2017.

[135] Hokeun Yoon, JinYeong Bak, *"Diversity Enhanced Narrative Question Generation for Storybooks"*, arXiv preprint:2310.16446v1, 2023.

[136] Sumithra Bhakthavatsalam, Daniel Khashabi, Tushar Khot, Bhavana Dalvi Mishra, Kyle Richardson, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord, Peter Clark, *"Think you have Solved Direct-Answer Question Answering? Try ARC-DA, the  Direct-Answer AI2 Reasoning Challenge"*, arXiv preprint:2102.03315v1, 2021.

[137] Yuxi Xie, Liangming Pan, Dongzhe Wang, Min-Yen Kan, Yansong Feng, *"Exploring Question-Specific Rewards for Generating Deep Questions"*, arXiv preprint:2011.01102v1, 2020.

[138] Hossein Bahak, Farzaneh Taheri, Zahra Zojaji, Arefeh Kazemi, *"Evaluating ChatGPT as a Question Answering System: A Comprehensive  Analysis and Comparison with Existing Models"*, arXiv preprint:2312.07592v1, 2023.

[139] Jingjing Li, Yifan Gao, Lidong Bing, Irwin King, Michael R. Lyu, *"Improving Question Generation With to the Point Context"*, arXiv preprint:1910.06036v2, 2019.

[140] Georgy Andryushchenko, Vladimir Ivanov, Vladimir Makharev, Elizaveta Tukhtina, Aidar Valeev, *"Leveraging Large Language Models in Code Question Answering: Baselines  and Issues"*, arXiv preprint:2411.03012v1, 2024.

[141] Esin Durmus, He He, Mona Diab, *"FEQA: A Question Answering Evaluation Framework for Faithfulness  Assessment in Abstractive Summarization"*, arXiv preprint:2005.03754v1, 2020.

[142] Deepak Gupta, Hardik Chauhan, Akella Ravi Tej, Asif Ekbal, Pushpak Bhattacharyya, *"Reinforced Multi-task Approach for Multi-hop Question Generation"*, arXiv preprint:2004.02143v4, 2020.

[143] Julian Risch, Timo Möller, Julian Gutsch, Malte Pietsch, *"Semantic Answer Similarity for Evaluating Question Answering Models"*, arXiv preprint:2108.06130v3, 2021.

[144] Nishant Balepur, Feng Gu, Abhilasha Ravichander, Shi Feng, Jordan Boyd-Graber, Rachel Rudinger, *"Reverse Question Answering: Can an LLM Write a Question so Hard (or Bad)  that it Can't Answer?"*, arXiv preprint:2410.15512v2, 2024.

[145] Xinghang Hu, *"Enhancing Answer Selection in Community Question Answering with  Pre-trained and Large Language Models"*, arXiv preprint:2311.17502v1, 2023.

[146] Siqing Huo, Negar Arabzadeh, Charles L. A. Clarke, *"Retrieving Supporting Evidence for LLMs Generated Answers"*, arXiv preprint:2306.13781v1, 2023.

[147] Dongjie Yang, Hai Zhao, *"Are LLMs Aware that Some Questions are not Open-ended?"*, arXiv preprint:2410.00423v1, 2024.

[148] Yunshi Lan, Xiang Li, Xin Liu, Yang Li, Wei Qin, Weining Qian, *"Improving Zero-shot Visual Question Answering via Large Language Models  with Reasoning Question Prompts"*, arXiv preprint:2311.09050v1, 2023.

[149] Qian Tao, Xiaoyang Fan, Yong Xu, Xingquan Zhu, Yufei Tang, *"Combining Knowledge Graph and LLMs for Enhanced Zero-shot Visual  Question Answering"*, arXiv preprint:2501.12697v1, 2025.

[150] Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, Chao Zhang, *"ToolQA: A Dataset for LLM Question Answering with External Tools"*, arXiv preprint:2306.13304v1, 2023.

[151] Zhihua Wen, Zhiliang Tian, Zexin Jian, Zhen Huang, Pei Ke, Yifu Gao, Minlie Huang, Dongsheng Li, *"Perception of Knowledge Boundary for Large Language Models through  Semi-open-ended Question Answering"*, arXiv preprint:2405.14383v1, 2024.

[152] Meghana Moorthy Bhat, Rui Meng, Ye Liu, Yingbo Zhou, Semih Yavuz, *"Investigating Answerability of LLMs for Long-Form Question Answering"*, arXiv preprint:2309.08210v1, 2023.

[153] Barah Fazili, Koustava Goswami, Natwar Modani, Inderjeet Nair, *"GenSco: Can Question Decomposition based Passage Alignment improve  Question Answering?"*, arXiv preprint:2407.10245v1, 2024.

[154] Chaojie Wang, Yishi Xu, Zhong Peng, Chenxi Zhang, Bo Chen, Xinrun Wang, Lei Feng, Bo An, *"keqing: knowledge-based question answering is a nature chain-of-thought  mentor of LLM"*, arXiv preprint:2401.00426v1, 2023.

[155] Jinheon Baek, Alham Fikri Aji, Amir Saffari, *"Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge  Graph Question Answering"*, arXiv preprint:2306.04136v1, 2023.

[156] Yunqi Zhu, Wen Tang, Ying Sun, Xuebing Yang, *"The Potential of LLMs in Medical Education: Generating Questions and  Answers for Qualification Exams"*, arXiv preprint:2410.23769v1, 2024.

[157] Lang Cao, *"Learn to Refuse: Making Large Language Models More Controllable and  Reliable through Knowledge Scope Limitation and Refusal Mechanism"*, arXiv preprint:2311.01041v4, 2023.

[158] Tiziano Labruna, Jon Ander Campos, Gorka Azkune, *"When to Retrieve: Teaching LLMs to Utilize Information Retrieval  Effectively"*, arXiv preprint:2404.19705v2, 2024.

[159] Parker Seegmiller, Joseph Gatto, Omar Sharif, Madhusudan Basak, Sarah Masud Preum, *"Do LLMs Find Human Answers To Fact-Driven Questions Perplexing? A Case  Study on Reddit"*, arXiv preprint:2404.01147v1, 2024.

[160] Xiaotian Lu, Jiyi Li, Koh Takeuchi, Hisashi Kashima, *"AHP-Powered LLM Reasoning for Multi-Criteria Evaluation of Open-Ended  Responses"*, arXiv preprint:2410.01246v1, 2024.

[161] Harry Li, Gabriel Appleby, Ashley Suh, *"LinkQ: An LLM-Assisted Visual Interface for Knowledge Graph  Question-Answering"*, arXiv preprint:2406.06621v2, 2024.

[162] Jinyoung Park, Ameen Patel, Omar Zia Khan, Hyunwoo J. Kim, Joo-Kyung Kim, *"Graph Elicitation for Guiding Multi-Step Reasoning in Large Language  Models"*, arXiv preprint:2311.09762v2, 2023.

[163] Yuyan Chen, Qiang Fu, Yichen Yuan, Zhihao Wen, Ge Fan, Dayiheng Liu, Dongmei Zhang, Zhixu Li, Yanghua Xiao, *"Hallucination Detection: Robustly Discerning Reliable Answers in Large  Language Models"*, arXiv preprint:2407.04121v1, 2024.

[164] Zixian Huang, Jiaying Zhou, Gengyang Xiao, Gong Cheng, *"Enhancing In-Context Learning with Answer Feedback for Multi-Span  Question Answering"*, arXiv preprint:2306.04508v1, 2023.

[165] Yifu Gao, Linbo Qiao, Zhigang Kan, Zhihua Wen, Yongquan He, Dongsheng Li, *"Two-stage Generative Question Answering on Temporal Knowledge Graph  Using Large Language Models"*, arXiv preprint:2402.16568v2, 2024.

[166] Alexander Bondarenko, Adrian Viehweger, *"LLM Robustness Against Misinformation in Biomedical Question Answering"*, arXiv preprint:2410.21330v1, 2024.

[167] Tairan Fu, Javier Conde, Gonzalo Martínez, María Grandury, Pedro Reviriego, *"Multiple Choice Questions: Reasoning Makes Large Language Models (LLMs)  More Self-Confident Even When They Are Wrong"*, arXiv preprint:2501.09775v2, 2025.

[168] Siqing Huo, Negar Arabzadeh, Charles L. A. Clarke, *"Retrieving Supporting Evidence for Generative Question Answering"*, arXiv preprint:2309.11392v1, 2023.

[169] Aryan Keluskar, Amrita Bhattacharjee, Huan Liu, *"Do LLMs Understand Ambiguity in Text? A Case Study in Open-world  Question Answering"*, arXiv preprint:2411.12395v1, 2024.

[170] Kai Sun, Yifan Ethan Xu, Hanwen Zha, Yue Liu, Xin Luna Dong, *"Head-to-Tail: How Knowledgeable are Large Language Models (LLMs)? A.K.A.  Will LLMs Replace Knowledge Graphs?"*, arXiv preprint:2308.10168v2, 2023.

[171] Camille Barboule, Benjamin Piwowarski, Yoan Chabot, *"Survey on Question Answering over Visually Rich Documents: Methods,  Challenges, and Trends"*, arXiv preprint:2501.02235v1, 2025.

[172] Debarshi Kundu, *"SciFaultyQA: Benchmarking LLMs on Faulty Science Question Detection with  a GAN-Inspired Approach to Synthetic Dataset Generation"*, arXiv preprint:2412.11988v1, 2024.

[173] Melissa Roemmele, Andrew S. Gordon, *"From Test-Taking to Test-Making: Examining LLM Authoring of Commonsense  Assessment Items"*, arXiv preprint:2410.14897v1, 2024.

[174] Jiejun Tan, Zhicheng Dou, Yutao Zhu, Peidong Guo, Kun Fang, Ji-Rong Wen, *"Small Models, Big Insights: Leveraging Slim Proxy Models To Decide When  and What to Retrieve for LLMs"*, arXiv preprint:2402.12052v3, 2024.

[175] Ankit Satpute, Noah Giessing, Andre Greiner-Petter, Moritz Schubotz, Olaf Teschke, Akiko Aizawa, Bela Gipp, *"Can LLMs Master Math? Investigating Large Language Models on Math Stack  Exchange"*, arXiv preprint:2404.00344v1, 2024.

[176] Aidar Myrzakhan, Sondos Mahmoud Bsharat, Zhiqiang Shen, *"Open-LLM-Leaderboard: From Multi-choice to Open-style Questions for LLMs  Evaluation, Benchmark, and Arena"*, arXiv preprint:2406.07545v1, 2024.

[177] Anirudh Phukan, Shwetha Somasundaram, Apoorv Saxena, Koustava Goswami, Balaji Vasan Srinivasan, *"Peering into the Mind of Language Models: An Approach for Attribution in  Contextual Question Answering"*, arXiv preprint:2405.17980v1, 2024.

[178] Jamshid Mozafari, Abdelrahman Abdallah, Bhawna Piryani, Adam Jatowt, *"Wrong Answers Can Also Be Useful: PlausibleQA -- A Large-Scale QA  Dataset with Answer Plausibility Scores"*, arXiv preprint:2502.16358v1, 2025.

[179] Hossein Rajabzadeh, Suyuchen Wang, Hyock Ju Kwon, Bang Liu, *"Multimodal Multi-Hop Question Answering Through a Conversation Between  Tools and Efficiently Finetuned Large Language Models"*, arXiv preprint:2309.08922v1, 2023.

[180] Junhao Chen, Bowen Wang, Zhouqiang Jiang, Yuta Nakashima, *"Putting People in LLMs' Shoes: Generating Better Answers via Question  Rewriter"*, arXiv preprint:2408.10573v2, 2024.

[181] Joshua Robinson, Christopher Michael Rytting, David Wingate, *"Leveraging Large Language Models for Multiple Choice Question Answering"*, arXiv preprint:2210.12353v3, 2022.

[182] Minh-Vuong Nguyen, Linhao Luo, Fatemeh Shiri, Dinh Phung, Yuan-Fang Li, Thuy-Trang Vu, Gholamreza Haffari, *"Direct Evaluation of Chain-of-Thought in Multi-hop Reasoning with  Knowledge Graphs"*, arXiv preprint:2402.11199v2, 2024.

[183] Yang Deng, Yong Zhao, Moxin Li, See-Kiong Ng, Tat-Seng Chua, *"Don't Just Say "I don't know"! Self-aligning Large Language Models for  Responding to Unknown Questions with Explanations"*, arXiv preprint:2402.15062v2, 2024.

[184] Eri Onami, Shuhei Kurita, Taiki Miyanishi, Taro Watanabe, *"JDocQA: Japanese Document Question Answering Dataset for Generative  Language Models"*, arXiv preprint:2403.19454v1, 2024.

[185] Siwei Wu, Xiangqing Shen, Rui Xia, *"A New Dialogue Response Generation Agent for Large Language Models by  Asking Questions to Detect User's Intentions"*, arXiv preprint:2310.03293v1, 2023.

[186] Xinyu Zhu, Cheng Yang, Bei Chen, Siheng Li, Jian-Guang Lou, Yujiu Yang, *"Question Answering as Programming for Solving Time-Sensitive Questions"*, arXiv preprint:2305.14221v3, 2023.

[187] Anishka IIITD, Diksha Sethi, Nipun Gupta, Shikhar Sharma, Srishti Jain, Ujjwal Singhal, Dhruv Kumar, *"TAMIGO: Empowering Teaching Assistants using LLM-assisted viva and code  assessment in an Advanced Computing Class"*, arXiv preprint:2407.16805v1, 2024.

[188] Marcos Fernández-Pichel, Juan C. Pichel, David E. Losada, *"Search Engines, LLMs or Both? Evaluating Information Seeking Strategies  for Answering Health Questions"*, arXiv preprint:2407.12468v2, 2024.

[189] Jieyu Zhang, Ranjay Krishna, Ahmed H. Awadallah, Chi Wang, *"EcoAssistant: Using LLM Assistant More Affordably and Accurately"*, arXiv preprint:2310.03046v1, 2023.

[190] Abhishek Kumar, Sonia Haiduc, Partha Pratim Das, Partha Pratim Chakrabarti, *"LLMs as Evaluators: A Novel Approach to Evaluate Bug Report  Summarization"*, arXiv preprint:2409.00630v1, 2024.

[191] Haopeng Zhang, Philip S. Yu, Jiawei Zhang, *"A Systematic Survey of Text Summarization: From Statistical Methods to  Large Language Models"*, arXiv preprint:2406.11289v1, 2024.

[192] Léo Hemamou, Mehdi Debiane, *"Scaling Up Summarization: Leveraging Large Language Models for Long Text  Extractive Summarization"*, arXiv preprint:2408.15801v1, 2024.

[193] Yixin Liu, Alexander R. Fabbri, Jiawen Chen, Yilun Zhao, Simeng Han, Shafiq Joty, Pengfei Liu, Dragomir Radev, Chien-Sheng Wu, Arman Cohan, *"Benchmarking Generation and Evaluation Capabilities of Large Language  Models for Instruction Controllable Summarization"*, arXiv preprint:2311.09184v2, 2023.

[194] Borui Xu, Yao Chen, Zeyi Wen, Weiguo Liu, Bingsheng He, *"Evaluating Small Language Models for News Summarization: Implications  and Factors Influencing Performance"*, arXiv preprint:2502.00641v2, 2025.

[195] Ankan Mullick, Sombit Bose, Rounak Saha, Ayan Kumar Bhowmick, Aditya Vempaty, Pawan Goyal, Niloy Ganguly, Prasenjit Dey, Ravi Kokku, *"Leveraging the Power of LLMs: A Fine-Tuning Approach for High-Quality  Aspect-Based Summarization"*, arXiv preprint:2408.02584v1, 2024.

[196] Dawit Mureja Argaw, Seunghyun Yoon, Fabian Caba Heilbron, Hanieh Deilamsalehy, Trung Bui, Zhaowen Wang, Franck Dernoncourt, Joon Son Chung, *"Scaling Up Video Summarization Pretraining with Large Language Models"*, arXiv preprint:2404.03398v1, 2024.

[197] Lionel Richy Panlap Houamegni, Fatih Gedikli, *"Evaluating the Effectiveness of Large Language Models in Automated News  Article Summarization"*, arXiv preprint:2502.17136v1, 2025.

[198] Shuaiqi Liu, Jiannong Cao, Yicong Li, Ruosong Yang, Zhiyuan Wen, *"Low-Resource Court Judgment Summarization for Common Law Systems"*, arXiv preprint:2403.04454v1, 2024.

[199] Lingxiao Wei, He Yan, Xiangju Lu, Junmin Zhu, Jun Wang, Wei Zhang, *"CNNSum: Exploring Long-Context Summarization with Large Language Models  in Chinese Novels"*, arXiv preprint:2412.02819v4, 2024.

[200] Zhenheng Tang, Xiang Liu, Qian Wang, Peijie Dong, Bingsheng He, Xiaowen Chu, Bo Li, *"The Lottery LLM Hypothesis, Rethinking What Abilities Should LLM  Compression Preserve?"*, arXiv preprint:2502.17535v1, 2025.

[201] Zirui Song, Bin Yan, Yuhan Liu, Miao Fang, Mingzhe Li, Rui Yan, Xiuying Chen, *"Injecting Domain-Specific Knowledge into Large Language Models: A  Comprehensive Survey"*, arXiv preprint:2502.10708v1, 2025.

[202] Ajay Jaiswal, Zhe Gan, Xianzhi Du, Bowen Zhang, Zhangyang Wang, Yinfei Yang, *"Compressing LLMs: The Truth is Rarely Pure and Never Simple"*, arXiv preprint:2310.01382v2, 2023.

[203] Zhijun Chen, Jingzheng Li, Pengpeng Chen, Zhuoran Li, Kai Sun, Yuankai Luo, Qianren Mao, Dingqi Yang, Hailong Sun, Philip S. Yu, *"Harnessing Multiple Large Language Models: A Survey on LLM Ensemble"*, arXiv preprint:2502.18036v1, 2025.

[204] Kung-Hsiang Huang, Philippe Laban, Alexander R. Fabbri, Prafulla Kumar Choubey, Shafiq Joty, Caiming Xiong, Chien-Sheng Wu, *"Embrace Divergence for Richer Insights: A Multi-document Summarization  Benchmark and a Case Study on Summarizing Diverse Information from News  Articles"*, arXiv preprint:2309.09369v2, 2023.

[205] Jiaan Wang, Yunlong Liang, Fandong Meng, Beiqi Zou, Zhixu Li, Jianfeng Qu, Jie Zhou, *"Zero-Shot Cross-Lingual Summarization via Large Language Models"*, arXiv preprint:2302.14229v4, 2023.

[206] Tingting Xu, Yun Miao, Chunrong Fang, Hanwei Qian, Xia Feng, Zhenpeng Chen, Chong Wang, Jian Zhang, Weisong Sun, Zhenyu Chen, Yang Liu, *"A Prompt Learning Framework for Source Code Summarization"*, arXiv preprint:2312.16066v2, 2023.

[207] Quanjun Zhang, Chunrong Fang, Yang Xie, Yaxin Zhang, Yun Yang, Weisong Sun, Shengcheng Yu, Zhenyu Chen, *"A Survey on Large Language Models for Software Engineering"*, arXiv preprint:2312.15223v2, 2023.

[208] Christopher T. Small, Ivan Vendrov, Esin Durmus, Hadjar Homaei, Elizabeth Barry, Julien Cornebise, Ted Suzman, Deep Ganguli, Colin Megill, *"Opportunities and Risks of LLMs for Scalable Deliberation with Polis"*, arXiv preprint:2306.11932v1, 2023.

[209] Jiangnan Fang, Cheng-Tse Liu, Jieun Kim, Yash Bhedaru, Ethan Liu, Nikhil Singh, Nedim Lipka, Puneet Mathur, Nesreen K. Ahmed, Franck Dernoncourt, Ryan A. Rossi, Hanieh Deilamsalehy, *"Multi-LLM Text Summarization"*, arXiv preprint:2412.15487v1, 2024.

[210] Qianqian Xie, Zheheng Luo, Benyou Wang, Sophia Ananiadou, *"A Survey for Biomedical Text Summarization: From Pre-trained to Large  Language Models"*, arXiv preprint:2304.08763v2, 2023.

[211] Xuanhe Zhou, Xinyang Zhao, Guoliang Li, *"LLM-Enhanced Data Management"*, arXiv preprint:2402.02643v1, 2024.

[212] Shivanshu Shekhar, Tanishq Dubey, Koyel Mukherjee, Apoorv Saxena, Atharv Tyagi, Nishanth Kotla, *"Towards Optimizing the Costs of LLM Usage"*, arXiv preprint:2402.01742v1, 2024.

[213] Aditi Godbole, Jabin Geevarghese George, Smita Shandilya, *"Leveraging Long-Context Large Language Models for Multi-Document  Understanding and Summarization in Enterprise Applications"*, arXiv preprint:2409.18454v1, 2024.

[214] Wensheng Gan, Zhenlian Qi, Jiayang Wu, Jerry Chun-Wei Lin, *"Large Language Models in Education: Vision and Opportunities"*, arXiv preprint:2311.13160v1, 2023.

[215] Tianyu Cao, Natraj Raman, Danial Dervovic, Chenhao Tan, *"Characterizing Multimodal Long-form Summarization: A Case Study on  Financial Reports"*, arXiv preprint:2404.06162v3, 2024.

[216] Yuvraj Virk, Premkumar Devanbu, Toufique Ahmed, *"Enhancing Trust in LLM-Generated Code Summaries with Calibrated  Confidence Scores"*, arXiv preprint:2404.19318v2, 2024.

[217] Shiqi Chen, Siyang Gao, Junxian He, *"Evaluating Factual Consistency of Summaries with Large Language Models"*, arXiv preprint:2305.14069v2, 2023.

[218] Jiuding Yang, Hui Liu, Weidong Guo, Zhuwei Rao, Yu Xu, Di Niu, *"SIFiD: Reassess Summary Factual Inconsistency Detection with LLM"*, arXiv preprint:2403.07557v1, 2024.

[219] Xuanliang Zhang, Dingzirui Wang, Longxu Dou, Qingfu Zhu, Wanxiang Che, *"A Survey of Table Reasoning with Large Language Models"*, arXiv preprint:2402.08259v1, 2024.

[220] Lemei Zhang, Peng Liu, Marcus Tiedemann Oekland Henriksboe, Even W. Lauvrak, Jon Atle Gulla, Heri Ramampiaro, *"PersonalSum: A User-Subjective Guided Personalized Summarization Dataset  for Large Language Models"*, arXiv preprint:2410.03905v1, 2024.

[221] Yiming Li, Fang Li, Kirk Roberts, Licong Cui, Cui Tao, Hua Xu, *"A Comparative Study of Recent Large Language Models on Generating  Hospital Discharge Summaries for Lung Cancer Patients"*, arXiv preprint:2411.03805v1, 2024.

[222] Zelalem Gero, Chandan Singh, Yiqing Xie, Sheng Zhang, Praveen Subramanian, Paul Vozila, Tristan Naumann, Jianfeng Gao, Hoifung Poon, *"Attribute Structuring Improves LLM-Based Evaluation of Clinical Text  Summaries"*, arXiv preprint:2403.01002v2, 2024.

[223] Fernando Gabriela Garcia, Spencer Burns, Harrison Fuller, *"Leveraging Large Language Models for Comparative Literature  Summarization with Reflective Incremental Mechanisms"*, arXiv preprint:2412.02149v1, 2024.

[224] Zheheng Luo, Qianqian Xie, Sophia Ananiadou, *"Factual Consistency Evaluation of Summarisation in the Era of Large  Language Models"*, arXiv preprint:2402.13758v1, 2024.

[225] Vladyslav Nechakhin, Jennifer D'Souza, Steffen Eger, *"Evaluating Large Language Models for Structured Science Summarization in  the Open Research Knowledge Graph"*, arXiv preprint:2405.02105v1, 2024.

[226] Veniamin Veselovsky, Manoel Horta Ribeiro, Philip Cozzolino, Andrew Gordon, David Rothschild, Robert West, *"Prevalence and prevention of large language model use in crowd work"*, arXiv preprint:2310.15683v1, 2023.

[227] Mayank Soni, Vincent Wade, *"Comparing Abstractive Summaries Generated by ChatGPT to Real Summaries  Through Blinded Reviewers and Text Classification Algorithms"*, arXiv preprint:2303.17650v3, 2023.

[228] Xianjun Yang, Yan Li, Xinlu Zhang, Haifeng Chen, Wei Cheng, *"Exploring the Limits of ChatGPT for Query or Aspect-based Text  Summarization"*, arXiv preprint:2302.08081v1, 2023.

[229] Zihao Yi, Jiarui Ouyang, Yuwen Liu, Tianhao Liao, Zhe Xu, Ying Shen, *"A Survey on Recent Advances in LLM-Based Multi-turn Dialogue Systems"*, arXiv preprint:2402.18013v1, 2024.

[230] Zhijian Chen, Chuan Hu, Min Wu, Qingqing Long, Xuezhi Wang, Yuanchun Zhou, Meng Xiao, *"GeneSUM: Large Language Model-based Gene Summary Extraction"*, arXiv preprint:2412.18154v1, 2024.

[231] Sanjana Ramprasad, Kundan Krishna, Zachary C Lipton, Byron C Wallace, *"Evaluating the Factuality of Zero-shot Summarizers Across Varied Domains"*, arXiv preprint:2402.03509v1, 2024.

[232] Ruvarashe Madzime, Clement Nyirenda, *"Enhanced Electronic Health Records Text Summarization Using Large  Language Models"*, arXiv preprint:2410.09628v1, 2024.

[233] Catarina G. Belem, Pouya Pezeskhpour, Hayate Iso, Seiji Maekawa, Nikita Bhutani, Estevam Hruschka, *"From Single to Multi: How LLMs Hallucinate in Multi-Document  Summarization"*, arXiv preprint:2410.13961v1, 2024.

[234] Laura Mascarell, Ribin Chalumattu, Annette Rios, *"German also Hallucinates! Inconsistency Detection in News Summaries with  the Absinth Dataset"*, arXiv preprint:2403.03750v2, 2024.

[235] Julia Evans, Jennifer D'Souza, Sören Auer, *"Large Language Models as Evaluators for Scientific Synthesis"*, arXiv preprint:2407.02977v1, 2024.

[236] Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, Xia Hu, *"Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond"*, arXiv preprint:2304.13712v2, 2023.

[237] Liyan Tang, Igor Shalyminov, Amy Wing-mei Wong, Jon Burnsky, Jake W. Vincent, Yu'an Yang, Siffi Singh, Song Feng, Hwanjun Song, Hang Su, Lijia Sun, Yi Zhang, Saab Mansour, Kathleen McKeown, *"TofuEval: Evaluating Hallucinations of LLMs on Topic-Focused Dialogue  Summarization"*, arXiv preprint:2402.13249v2, 2024.

[238] Rongxin Zhu, Jey Han Lau, Jianzhong Qi, *"Factual Dialogue Summarization via Learning from Large Language Models"*, arXiv preprint:2406.14709v1, 2024.

[239] Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, Ajmal Mian, *"A Comprehensive Overview of Large Language Models"*, arXiv preprint:2307.06435v10, 2023.

