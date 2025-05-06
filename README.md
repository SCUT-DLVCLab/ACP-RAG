# ACP-RAG
[NAACL 2025] Large-Scale Corpus Construction and Retrieval-Augmented Generation for Ancient Chinese Poetry: New Method and Data Insights

## Introduction
**Ancient Chinese Poetry (ACP)**, a critical aspect of Chinese cultural heritage, presents unique challenges for Large Language Models (LLMs), primarily due to significant hallucination issues arising from data scarcity and the limited ability of general LLMs to handle ACP. To address these challenges, this paper introduces the **ACP-Corpus**, comprising 1.1 million ancient poems and 990K related texts, designed to enhance LLM training and performance. Additionally, we develop the **ACP-QA** dataset, containing over 12 million question-answer pairs across 24 task categories, and the **ACP-Eval** dataset with 7,050 entries for rigorous evaluation. Building upon these resources, we propose the **ACP-RAG** framework, a specialized Retrieval-Augmented Generation (RAG) approach that improves LLM performance in the ACP domain from 49.2% to 89.0%. The ACP-RAG consists of five modules: **semantic coarse-grained retrieval, semantic fine-grained retrieval, keyword retrieval, keyword matching, and context filtering**. Experimental results demonstrate that ACP-RAG achieves a promising response accuracy of 89.0%, surpassing existing LLMs by a remarkable margin. This work not only advances the capabilities of LLMs in processing ancient Chinese poetry but also contributes to the preservation and innovative development of this rich literary tradition.

<div align="center">
  <img src="Images/Figure1.png" width="90%" />
</div>

<div align="center">
Figure 1: Overview of the ACP-RAG framework. Model A is the embedding model, Model B is the rank model, and Model C is the keyword extraction and context filtering model (Qwen1.5-7B). Zoom in for better view.
</div>

## Task Description

<div align="center">
Table 1: Description of 24 ancient Chinese poetry tasks.
</div>

<div align="center">
  <img src="Images/Table5.png" width="90%" />
</div>

## ACP-Corpus

<div align="center">
Table 2: Comparison of existing pre-training corpora. “ET” indicates English Translation, “PA” indicates Poem Appreciation, “VT” indicates Vernacular Translation, “WE” indicates Word Explanation, “ID” indicates Idioms, “LK” indicates Literary Knowledge, and “PI” indicates Poet Introduction.
</div>

<div align="center">
  <img src="Images/Table1.png" width="90%" />
</div>

## ACP-QA

<div align="center">
Table 3: Comparison of existing Q&A datasets. “IC” indicates Instruction Categories, “HG” indicates Human Generated, “CI” indicates Collection and Improvement of existing datasets, and “MC” indicates Model Constructed.
</div>

<div align="center">
  <img src="Images/Table2.png" width="90%" />
</div>

## Evaluation Metrics

<div align="center">
  <img src="Images/Figure2.png" width="50%" />
</div>

<div align="center">
Figure 2: RAG evaluation system.
</div>

## Result

<div align="center">
Table 4: Comparison between ACP-RAG and other methods on ACP-Eval. “RA” indicates Response Accuracy, “RC” indicates Response Continuity, “RR” indicates Response Relevance, “CIV” indicates Context Information Volume, “CMS” indicates Context Match Score, and “CTR” indicates Context Topic Relevance.
</div>

<div align="center">
  <img src="Images/Table4.png" width="60%" />
</div>

## Citation

```
@article{liu2025acprag,
  title={Large-Scale Corpus Construction and Retrieval-Augmented Generation for Ancient Chinese Poetry: New Method and Data Insights},
  author={Liu, Yang and Lan, Lan and Cao, Jiahuan and Cheng, Hiuyi and Ding, Kai and Jin, Lianwen},
  journal={NAACL 2025},
  year={2025}
}
```

## License

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

The work is licensed under a [MIT License](https://lbesson.mit-license.org/).

![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)

The datasets are licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Contact

Yang Liu: ly10061105@gmail.com




