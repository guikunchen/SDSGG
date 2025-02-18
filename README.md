# SDSGG
This is the code for our NeurIPS 2024 paper "[Scene Graph Generation with Role-Playing Large Language Models](https://arxiv.org/abs/2410.15364)".

## Abstract
Current approaches for open-vocabulary scene graph generation (OVSGG) use vision-language models such as CLIP and follow a standard zero-shot pipeline -- computing similarity between the query image and the text embeddings for each category (i.e., text classifiers). In this work, we argue that the text classifiers adopted by existing OVSGG methods, i.e., category-/part-level prompts, are scene-agnostic as they remain unchanged across contexts. Using such fixed text classifiers not only struggles to model visual relations with high variance, but also falls short in adapting to distinct contexts. To plug these intrinsic shortcomings, we devise SDSGG, a scene-specific description based OVSGG framework where the weights of text classifiers are adaptively adjusted according to the visual content. In particular, to generate comprehensive and diverse descriptions oriented to the scene, an LLM is asked to play different roles (e.g., biologist and engineer) to analyze and discuss the descriptive features of a given scene from different views. Unlike previous efforts simply treating the generated descriptions as mutually equivalent text classifiers, SDSGG is equipped with an advanced renormalization mechanism to adjust the influence of each text classifier based on its relevance to the presented scene (this is what the term "specific" means). Furthermore, to capture the complicated interplay between subjects and objects, we propose a new lightweight module called mutual visual adapter. It refines CLIP's ability to recognize relations by learning an interaction-aware semantic space. Extensive experiments on prevalent benchmarks show that SDSGG outperforms top-leading methods by a clear margin.

## Scene Graph Generation
### 1. Installation

Check INSTALL.md for installation instructions.

### 2. Data 

We conducted evaluations on the VG and GQA datasets.

```
│darasers/
├──gqa/
│  ├── images
│  ├── ......
├──vg/
│  ├── VG100K
│  ├── ......
```

### 3. Scripts

**See script.txt.**

### 4. Checkpoints

We uploaded all the checkpoints to [BaiduDrive](https://pan.baidu.com/s/1YJXutOUNx74ac1A0Lu5NSQ?pwd=ywhx). Feel free to download.


## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[Scene Graph Benchmark in Pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), [RECODE](https://github.com/HKUST-LongGroup/RECODE).

## Citation

If you find this work useful in your research, please star our repository and consider citing:

```
@inproceedings{chen2024scene,
  title={Scene Graph Generation with Role-Playing Large Language Models},
  author={Chen, Guikun and Li, Jin and Wang, Wenguan},
  booktitle={NeurIPS},
  year={2024}
}
```

## Contact

Any comments, please email: guikunchen@gmail.com.

