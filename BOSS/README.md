"Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance"
## Accepted to CoRL 2023 as an Oral Presentation (top 6.6%).


[[Project Website]](https://clvrai.github.io/boss/) [[Paper]](https://arxiv.org/abs/2310.10021) [[OpenReview]](https://openreview.net/forum?id=a0mFRgadGO)

[Jesse Zhang](https://jesbu1.github.io/)<sup>1</sup>, [Jiahui Zhang](https://jiahui-3205.github.io/)<sup>1</sup>, [Karl Pertsch](https://kpertsch.github.io/)<sup>1</sup>, [Ziyi Liu](https://taichi-pink.github.io/Ziyi-Liu/)<sup>1</sup>, [Xiang Ren](https://shanzhenren.github.io/)<sup>1</sup>, [Minsuk Chang](https://minsukchang.com/)<sup>2</sup>, [Shao-Hua Sun](https://shaohua0116.github.io/)<sup>3</sup>, [Joseph J. Lim](https://clvrai.com/web_lim/)<sup>4</sup>

<sup>1</sup>University of Southern California 
<sup>2</sup>Google Deepmind
<sup>3</sup>National Taiwan University
<sup>4</sup>KAIST

This is the official PyTorch implementation of CoRL 2023 paper "**Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance**"

## Overview


<a href="images/boss_overview.png">
<p align="center">
<img src="images/boss_overview.png" width="800">
</p>
</img></a>


#### Paper description and Main Idea:
We propose BOSS, an approach that automatically learns to solve new long-horizon, complex, and meaningful tasks by growing a learned skill library with minimal supervision. Prior work in reinforcement learning require expert supervision, in the form of demonstrations or rich reward functions, to learn long-horizon tasks. Instead, our approach BOSS (BOotStrapping your own Skills) learns to accomplish new tasks by performing "skill bootstrapping," where an agent with a set of primitive skills interacts with the environment to practice new skills without receiving reward feedback for tasks outside of the initial skill set. This bootstrapping phase is guided by large language models (LLMs) that inform the agent of meaningful skills to chain together. Through this process, BOSS builds a wide range of complex and useful behaviors from a basic set of primitive skills. We demonstrate through experiments in realistic household environments that agents trained with our LLM-guided bootstrapping procedure outperform those trained with naive bootstrapping as well as prior unsupervised skill acquisition methods on zero-shot execution of unseen, long-horizon tasks in new environments.

#### Contribution:
BOSS Contributes a method which
* Utilizes LLM-guided "skill bootstrapping" to allow a closed-loop, learned agent to fine-tune and learn new complex behaviors without prior task guidance
* Results in an instruction-following policy which does not require an LLM at test time to execute language instructions at various levels of abstraction.

## Environment Setup
We include a conda `environment.yml`, please use that to install the environment.
We use WandB, you will need to create a [wandb account](wandb.ai) and change the `WANDB_PROJECT_NAME` and `WANDB_ENTITY_NAMES` at the top of the files you run.

## Training and Evaluation

### 1) Pretrained checkpoints and data
We release our pre-trained base model checkpoints [here](https://drive.google.com/file/d/1TYaTLf1t8CwaYzUirQIwXBQ4_GdQAeZn/view?usp=drive_link).
Our processed ALFRED dataset is [here](https://drive.google.com/file/d/1ZgKDgG9Fv491GVb9rxIVNJpViPNKFWMF/view?usp=drive_link).

You should be able to download these with [gdown](https://github.com/wkentaro/gdown) or manually.

To pre-train a model yourself, you need the ALFRED dataset and you can then run: 

```shell script
python alfred_et_iql.py --gpus [GPUID] --advanced_critics True --experiment_name [NAME] --seed [SEED] --run_group boss_pretrain_model
```

### 2) Skill Bootstrapping and Evaluation commands
You will need sufficient GPU resources to run skill bootstrapping. Ai2Thor/ALFRED will also put all environment instances on GPU 0. 
We typically leave GPU 0 empty for ALFRED and then use a separate 3090 GPU to run skill bootstrapping w/ the LLM and training simultaneously.

There are 4 floorplans in ALFRED, 0 through 3, we run bootstrapping on each floorplan (each with 10 tasks) separately.
You can modify the floorplan ID in the script below to train on that floorplan.

```skill bootstrapping script
CUDA_VISIBLE_DEVICES=[LLM GPU] python alfred_rl_online_et.py --llm_model decapoda-research/llama-13b-hf --llm_batch_size 2 --run_group [GROUP_NAME] --load_model_path [SAVED MODEL LOC] --which_floorplan [FLOORPLAN ID]

```

## Reference

If you find the code useful for your research, please consider citing
```bib
@inproceedings{
    zhang2023bootstrap,
    title={Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance},
    author={Jesse Zhang and Jiahui Zhang and Karl Pertsch and Ziyi Liu and Xiang Ren and Minsuk Chang and Shao-Hua Sun and Joseph J Lim},
    booktitle={7th Annual Conference on Robot Learning},
    year={2023},
    url={https://openreview.net/forum?id=a0mFRgadGO}
}
```
