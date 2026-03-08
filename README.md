<div align="center">


<img src="./assets/logo.png" width="500em" ></img> 

**An Open-Source Framework for
Scaling Generative Recommendation**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)
<a href="https://arxiv.org/abs/2510.24431"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red"></a>

<a href="https://arxiv.org/abs/2510.24431">📄 Technical Report</a> | <a href="https://huggingface.co/kkknight/MiniOneRec">🤗 Huggingface</a> | <a href="https://modelscope.cn/models/k925238839/MiniOneRec">🤖  Modelscope</a>
</div>

**MiniOneRec** is the first fully open-source **generative recommendation** framework, which provides an end-to-end workflow spanning **SID construction**, **supervised fine-tuning (SFT)**, and recommendation-oriented **reinforcement learning (RL)**. 

---

## 📢 Announcement

- 2026-01-04 — Regarding the potential discrepancies between the reproduced results based on the Instruct model and our reported metrics, please check whether the CC metric in the evaluation log is non-zero (refer to utils/calc.py). If it is non-zero, it indicates that the model is still generating a large number of invalid items, and constrained decoding has not been successful. We suspect this issue may be related to the versions of dependencies such as the transformer library, and we are still investigating the cause to provide a universal solution. In the meantime, you may switch the Instruct model to a base model, such as Qwen2.5-base, to avoid this problem.

- 2025-12-04 — We update new scripts to support processing the Amazon23 dataset.

- 2025-12-01 — We fix a bug in utils/data.py that could cause the SID–item alignment task to see the answers in advance. This was because we had previously attempted to use partial trajectories to guide the full SID–item generation and does not affect the model performance.

- 2025-11-20 — The SID construction method in **RQ-Kmeans+** has been updated (first proposed in **GPR** and this is the first open-source reproduction).

- 2025-11-19 — We implemented a multi-GPU parallel text-to-embedding method based on Accelerate, which is significantly more efficient than the original version: rq/text2emb/amazon_text2emb.py

- 2025-11-19 — The SID construction method in **constrained-RQ-Kmeans** has been updated.

- 2025-11-07 — Thank you for submitting issues! Based on your feedback, we have released a new implementation. If you encounter any problems while running the code, please update to and consult the **latest version** first.
  
- 2025-11-07 — You can now choose to freeze the LLM parameters during the SFT stage and train only the embeddings for the newly added SID vocabulary.

- 2025-10-31 — You can now directly download the implementation **checkpoints** of our MiniOnRec model.

- 2025-10-31 — The SID construction method in **RQ-Kmeans** has been updated.

---

## 🛠️ Key Techniques 
<div align="center">
<img src="./assets/minionerec_framework.png" width=100% ></img> 
</div>

- **SID Construction: MiniOneRec begins by transforming every product into a compact, semantically meaningful token.** It concatenates an item’s title and description, feeds this sentence through a frozen text encoder, and then quantises the resulting embedding with a three-level RQ-VAE.

- **SFT: With all items rewritten as SIDs, the model is first trained in a supervised fashion.** It views the chronologically ordered user history as a token sequence and learns, via next-token prediction, to generate the SID of the next product the user is likely to consume. Crucially, this stage is co-trained with a set of language-alignment objectives that map back and forth between natural language and SID space, allowing the recommender to inherit the world knowledge embedded in large language models while grounding that knowledge in discrete item codes.

- **Recommendation-Oriented RL: After SFT, MiniOneRec is further polished with a recommendation-oriented RL phase based on GRPO.** Multiple candidate recommendations are generated for each prompt, their rewards are normalised within the group to stabilise gradients, and a KL penalty keeps the updated policy close to its reference. Because the action space is a closed list of item SIDs, the system switches to constrained beam search, which guarantees that every beam is unique and valid, greatly improving sampling efficiency and diversity. The reward signal itself blends a binary correctness term with a rank-aware component that penalises high-probability yet incorrect items more heavily, and can be augmented with collaborative-filtering scores. Together, this pipeline enables MiniOneRec to couple dense linguistic knowledge, achieving a high-performance, lightweight generative recommendation system.

---

## 📊 Evaluation

<div align="center">
<img src="./assets/minionerec_main_result.png" width=100% ></img>
</div>

### Reproduction Results (Industrial_and_Scientific, Qwen3.5-2B)

#### Accuracy Metrics

| Stage | HR@1 | HR@5 | HR@10 | HR@50 | NDCG@5 | NDCG@10 | NDCG@50 | MRR@10 | MRR@50 |
|-------|------|------|-------|-------|--------|---------|---------|--------|--------|
| Base Model (no SFT) | 1.96% | 5.56% | 8.43% | 16.48% | 3.89% | 4.79% | 6.64% | 3.69% | 4.12% |
| SFT | 4.06% | 8.07% | 10.41% | 18.44% | 6.18% | 6.94% | 8.70% | 5.86% | 6.24% |
| RL | — | — | — | — | — | — | — | — | — |

#### Diversity Metrics

| Stage | Coverage@10 | Coverage@50 | ILS@10 | Entropy@10 | Entropy@50 |
|-------|-------------|-------------|--------|------------|------------|
| Base Model (no SFT) | 96.66% | 98.78% | 0.2000 | 11.0007 | 10.8147 |
| SFT | 66.58% | 89.69% | 0.1358 | 9.6386 | 10.3452 |
| RL | — | — | — | — | — |

#### Fairness Metrics

| Stage | Gini@10 | Gini@50 | LongTail-Cov@10 | LongTail-Cov@50 |
|-------|---------|---------|------------------|-----------------|
| Base Model (no SFT) | 0.5251 | 0.6193 | 96.06% | 98.94% |
| SFT | 0.7175 | 0.6908 | 56.76% | 86.53% |
| RL | — | — | — | — |

#### Novelty Metrics

| Stage | Novelty@10 | Novelty@50 |
|-------|------------|------------|
| Base Model (no SFT) | 12.2791 | 12.6547 |
| SFT | 11.1337 | 11.4989 |
| RL | — | — |

> Max possible entropy: 11.8478 | Long-tail items: 1878 (threshold: pop <= 0.000162) | Novelty = Mean Self-Information in bits (higher = more novel)

#### Analysis: Base Model vs. SFT

SFT yields substantial accuracy gains across all metrics, with the most pronounced improvements at small K values (HR@1: +107%, NDCG@5: +59%, MRR@10: +59%), indicating that the model learns to rank the ground-truth item significantly higher after fine-tuning.

However, a classic **accuracy–diversity trade-off** is observed: Coverage@10 drops from 96.66% to 66.58%, Entropy@10 decreases by ~1.4 bits, the Gini coefficient rises from 0.5251 to 0.7175, and LongTail-Cov@10 falls sharply from 96.06% to 56.76%. This is expected — the base model produces near-uniform predictions over the item space, while SFT concentrates probability mass on a smaller set of popular items.

Novelty also decreases (12.28 → 11.13 bits at @10), confirming that SFT shifts recommendations toward more commonly interacted items.

**Takeaway:** SFT is effective at improving recommendation accuracy but introduces popularity bias and reduces diversity. This motivates the subsequent **RL stage (GRPO)**, which is designed to recover diversity and fairness while maintaining accuracy gains.

#### Analysis: SFT vs. RL (TODO)

> *RL results pending. Expected effects: further accuracy improvement with partial recovery of diversity and long-tail coverage, driven by the ranking reward and constrained beam search in GRPO.*

---

## 🚀 Quickstart

Use the pre-trained Industrial/Office SIDs we provide for a quick start!
Reproduction can be achieved with just 4–8 A100/H100 GPUs.

### 1. Create an isolated Python environment

```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. SFT

```bash
bash scripts/sft.sh
```

### 4. Recommendation-Oriented RL

```bash
bash scripts/rl.sh
```

### 5. Run the evaluation bash

```bash
bash scripts/evaluate.sh
```

---

## 📜 Full Pipeline Walk-through

### 0. Prerequisites
- GPUs: <e.g., 4–8 × A100/H100 80 GB or comparable>
- Python: 3.11

### 1. Environment Setup
- **1.1 Clone the repo**
```
git clone https://github.com/AkaliKong/MiniOneRec.git
cd MiniOneRec
```
- **1.2 Create and activate a conda env**
```
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```
- **1.3 Install dependencies**
```
pip install -r requirements.txt
```

### 2. Data Preparation

- **2.1 Download the raw dataset (Optional)**  
  Get it from the official page:
  [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), 
  [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), 
  [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
  Note: The Industrial and Office datasets are included in Amazon 2018; the Amazon 2014 and 2023 versions require slight modifications to our data/amazon18_data_process.py.
- **2.2 Filter and preprocess**
```
bash data/amazon18_data_process.sh \
     --dataset  your_dataset_type \ # e.g. Industrial
     --user_k 5 \
     --item_k 5 \
     --st_year 2017 \
     --st_month 10 \
     --ed_year 2018 \
     --ed_month 11 \
     --output_path ./data/Amazon18
```
- **2.3 Encode item text to embeddings**
```
bash rq/text2emb/amazon_text2emb.sh \
     --dataset your_dataset_type \ # e.g., Industrial 
     --root your_processed_dataset_path \
     --plm_name qwen \
     --plm_checkpoint your_emb_model_path
```

### 3. SID Construction

Choose either 3.1.1, 3.1.2, 3.1.3 or 3.1.4.

- **3.1.1 Train RQ-VAE on the embeddings**
```
bash rq/scripts/rqvae.sh \
      --data_path xxx/data/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
      --ckpt_dir ./output/Industrial_and_Scientific \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480
```

- **3.1.2 Train RQ-Kmeans on the embeddings**

```
conda install faiss-gpu
python rq/trainer/rqkmeans_faiss.py --dataset Industrial_and_Scientific # The RQ-Kmeans method based on semantic embeddings has a relatively high collision rate.
```

- **3.1.3 Train constrained RQ-Kmeans on the embeddings**
For conflicting items, we add an extra layer to perform deduplication; meanwhile, we use a balanced constraint to ensure that the SIDs are evenly distributed.
```
pip install k_means_constrained
pip install polars
bash rq/scripts/rqkmeans_constrained.sh
```

- **3.1.4 Train RQ-Kmeans+ on the embeddings**
```
pip install k_means_constrained
pip install polars
bash rq/scripts/rqkmeans_constrained.sh
bash rq/scripts/rqkmeans_plus.sh
```

- **3.2 Generate indices(only RQ-VAE & RQ-Kmeans+ needed)**
```
python rq/trainer/generate_indices.py
# or
bash rq/scripts/generate_indices_plus.sh
```

- **3.3 Convert dataset format**
```
python utils/convert_dataset.py \
     --dataset_name Industrial_and_Scientific \
     --data_dir /path/to/Industrial_and_Scientific \
     --output_dir /path/to/ourput_dir \

```

### 4. SFT

Edit `scripts/sft.sh` to set your category and base model, then run:
```bash
bash scripts/sft.sh
```

The key parameters inside the script:
```bash
CUDA_VISIBLE_DEVICES=0 python trainer/sft.py \
    --base_model Qwen/Qwen3.5-2B \
    --batch_size 128 \
    --micro_batch_size 16 \
    --train_file ${train_file} \
    --eval_file ${eval_file} \
    --output_dir ./output/sft/${category} \
    --wandb_project MiniOneRec \
    --wandb_run_name sft_${category}_qwen3.5-2b \
    --category ${category} \
    --train_from_scratch False \
    --seed 42 \
    --sid_index_path ./data/Amazon/index/${category}.index.json \
    --item_meta_path ./data/Amazon/index/${category}.item.json \
    --freeze_LLM False
```

| Parameter | Description |
|-----------|-------------|
| `base_model` | HuggingFace model name or local path |
| `batch_size` / `micro_batch_size` | Total batch size and per-device batch size (gradient accumulation = batch_size / micro_batch_size) |
| `sid_index_path` | Path to `.index.json` generated by SID construction |
| `item_meta_path` | Path to `.item.json` containing item metadata |
| `freeze_LLM` | If `True`, freeze all LLM parameters and only train new SID token embeddings |
| `train_from_scratch` | If `True`, initialize model weights randomly instead of loading pretrained weights |

### 5. Recommendation-Oriented RL
> (Optional) For production-scale datasets, considering the cost of reinforcement learning and diminishing marginal returns, you can perform the RL stage using only a relatively small subset on the order of tens of thousands of samples.

Edit `scripts/rl.sh` to set your category and model path, then run:
```bash
bash scripts/rl.sh
```

The key parameters inside the script:
```bash
CUDA_VISIBLE_DEVICES=0 python trainer/rl.py \
    --model_path ./output/sft/${category} \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 4 \
    --train_file ${train_file} \
    --eval_file ${eval_file} \
    --info_file ${info_file} \
    --category ${category} \
    --reward_type ranking \
    --num_generations 16 \
    --sync_ref_model True \
    --beam_search True \
    --temperature 1.0 \
    --learning_rate 1e-5 \
    --beta 1e-3 \
    --output_dir ./output/rl/${category} \
    --wandb_run_name rl_${category}_qwen3.5-2b \
    --sid_index_path ./data/Amazon/index/${category}.index.json \
    --item_meta_path ./data/Amazon/index/${category}.item.json
```

| Parameter | Description |
|-----------|-------------|
| `model_path` | Path to the SFT checkpoint (e.g. `./output/sft/${category}`) |
| `reward_type` | Reward function: `rule` (binary), `ranking` (rule + NDCG-aware), `sasrec` (CF-based), `semantic` (embedding similarity) |
| `num_generations` | Number of candidate completions per prompt in GRPO |
| `beam_search` | Use constrained beam search for generating valid SIDs |
| `sync_ref_model` | Periodically sync the reference model with the policy model |
| `beta` | KL penalty coefficient |

### 6. Offline Evaluation

Edit `scripts/evaluate.sh` to set `exp_name` (model path) and category, then run:
```bash
bash scripts/evaluate.sh
```

The key parameters inside the script:
```bash
CUDA_VISIBLE_DEVICES=0 python -u ./evaluate.py \
    --base_model "$exp_name" \
    --info_file "$info_file" \
    --category ${category} \
    --test_data_path "$test_file" \
    --result_json_data "$result_dir/result_${category}.json" \
    --batch_size 4 \
    --num_beams 50 \
    --max_new_tokens 256 \
    --length_penalty 0.0

# Calculate metrics
python ./utils/calc.py \
    --path "$result_dir/result_${category}.json" \
    --item_path "$info_file"
```

| Parameter | Description |
|-----------|-------------|
| `exp_name` | Path to model checkpoint (SFT or RL output) |
| `num_beams` | Number of beams for constrained beam search (Top-K) |
| `batch_size` | Evaluation batch size (reduce if OOM) |
| `max_new_tokens` | Maximum generated tokens per sample |

---

## 📝 Upcoming Features

We are actively extending MiniOneRec’s capabilities. The following enhancements are already on our roadmap:
* ⏱️ **More SID Construction Algorithms**: forthcoming support for R-VQ, RQ-Kmeans, RQ-OPQ, and RQ-VAE-v2 (PLUM).
* ⚙️ **MiniOneRec-Think**: a module that seamlessly integrates dialogue, reasoning, and personalized recommendation, providing an all-in-one solution for complex interactive scenarios.
* 🔍 **Broader Dataset Support**: additional popular public datasets, including Yelp, to further validate the generality of our algorithms.

---

## 🏫 Institutions  <!-- omit in toc -->

This project is developed by the following institutions:

- <img src="assets/lds.png" width="28px"> [LDS](https://data-science.ustc.edu.cn/_upload/tpl/15/04/5380/template5380/index.html)
- <img src="assets/alphalab.jpg" width="28px"> [AlphaLab](https://alphalab-ustc.github.io/index.html)
- <img src="assets/next.jpg" width="28px"> [NExT](https://www.nextcenter.org/)
 
---

## 🧩 Contributing

We welcome and appreciate all contributions! If you have ideas to improve MiniOneRec, please feel free to submit a pull request (PR).

---
## 🙏 Acknowledgements

This repository reuses or adapts portions of code from the following open-source projects. We gratefully acknowledge their authors and contributors:

- [ReRe](https://github.com/sober-clever/ReRe)
- [LC-Rec](https://github.com/zhengbw0324/LC-Rec)

---

## 🔖 Citation <!-- omit in toc -->

If you find our code/paper/model helpful, please consider citing our papers 📝 and staring us ⭐️！

```bib
@misc{MiniOneRec,
      title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation}, 
      author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
      year={2025},
      eprint={2510.24431},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
}

@article{ReRe,
      title={Reinforced Preference Optimization for Recommendation}, 
      author={Junfei Tan and Yuxin Chen and An Zhang and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Xiang Wang},
      journal={arXiv preprint arXiv:2510.12211},
      year={2025},
}

@inproceedings{RecZero,
      title={Think before Recommendation: Autonomous Reasoning-enhanced Recommender}, 
      author={Xiaoyu Kong and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Jiancan Wu and Xiang Wang},
      year={2025},
      booktitle={NeurIPS},
}

```

---

<div align="center">
We welcome contributions from the community! 🤝
</div>
