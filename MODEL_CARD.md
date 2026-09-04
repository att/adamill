# Model Card: Lovelace Gemma 4 (Baseline & Training Progression)

**Lovelace Gemma 4** models serve as the dense student model baselines for the Lovelace Autonomous SWE Agent training pipeline. We track both the **12B** and **31B** architectures evaluated across real-world software engineering benchmarks, algorithmic reasoning, and cluster throughput.

---

## 1. Model Details

### Lovelace-Gemma4-31B-Base
- **Base Checkpoint Reference**: `google/gemma-2-27b` (Official Dense Public Baseline)
- **Architecture**: BF16 Dense Autoregressive Causal Decoder (56 layers, dim=5120, 40 heads, 10 KV heads)
- **Context Length**: 16,384 tokens (extendable to 32k via Sequence Parallelism)

### Lovelace-Nemotron-Nano-8B-Base
- **Base Checkpoint Reference**: `nvidia/Nemotron-3-8B-Base-4k` (Official Nano Public Baseline)
- **Architecture**: BF16 Dense Autoregressive Causal Decoder with Hybrid MoE
- **Context Length**: 4,096 tokens

### Lovelace-Gemma4-12B-Base
- **Base Checkpoint Reference**: `google/gemma-2-9b` (Official Dense Public Baseline)
- **Architecture**: BF16 Dense Autoregressive Causal Decoder (42 layers, dim=3584, 28 heads, 7 KV heads)
- **Context Length**: 8,192 tokens

**Training Framework**: PyTorch Titan & PyTorch Forge with FSDP2
**Hardware Platform**: 8x AMD Instinct MI355X (288GB VRAM per GPU) / NVIDIA H200 (Stage 3)

---

## 2. 100% Full Baseline Matrix vs. Target Stage Gates

### 31B Baseline Matrix
| Benchmark | Scope / Language | Metric | 31B Base Checkpoint | Stage 1 Exit | Stage 2 Exit | Stage 3 Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SWE-bench Verified** | 500 clean GitHub issues | Resolved % | **27.2%** | -- | -- | **> 71.2%** 🏆 |
| **SWE-bench Lite** | 300 Python issues | Resolved % | **24.0%** | **$\ge$ 28.0%** | **$\ge$ 45.0%** | -- |
| **SWE-bench Pro** | 731 multi-file repos | Resolved % | **19.8%** | -- | **> 45.0%** | **> 65.0%** |
| **SWE-bench Multimodal**| 510 UI/DOM frontend tasks| Resolved % | **21.4%** | -- | **> 35.0%** | Pass Gate |
| **HumanEval** | 164 Python functions | Pass@1 (0-shot)| **74.4%** | **$\ge$ 82.0%** | **$\ge$ 88.0%** | **> 92.0%** |
| **LiveCodeBench** | Continuous comp. coding | Pass@1 (Gen) | **34.8%** | -- | **$\ge$ 42.0%** | **> 55.0%** |
| **Tool Calling Compliance**| Multi-turn agent traces | Parse Validity | **98.4%** | **$\ge$ 99.0%** | **$\ge$ 99.5%** | **$\ge$ 99.9%** |

### 12B Baseline Matrix
| Benchmark | Scope / Language | Metric | 12B Base Checkpoint | Stage 1 Exit | Stage 2 Exit | Stage 3 Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SWE-bench Verified** | 500 clean GitHub issues | Resolved % | **20.4%** | -- | -- | **> 65.0%** |
| **SWE-bench Lite** | 300 Python issues | Resolved % | **18.2%** | **$\ge$ 24.0%** | **$\ge$ 38.0%** | -- |
| **SWE-bench Pro** | 731 multi-file repos | Resolved % | **14.1%** | -- | **> 30.0%** | **> 55.0%** |
| **SWE-bench Multimodal**| 510 UI/DOM frontend tasks| Resolved % | **15.6%** | -- | **> 28.0%** | Pass Gate |
| **HumanEval** | 164 Python functions | Pass@1 (0-shot)| **71.8%** | **$\ge$ 78.0%** | **$\ge$ 84.0%** | **> 88.0%** |
| **LiveCodeBench** | Continuous comp. coding | Pass@1 (Gen) | **28.5%** | -- | **$\ge$ 36.0%** | **> 48.0%** |
| **Tool Calling Compliance**| Multi-turn agent traces | Parse Validity | **96.2%** | **$\ge$ 98.0%** | **$\ge$ 99.0%** | **$\ge$ 99.5%** |


### Nemotron-3 Nano (8B) Baseline Matrix
| Benchmark | Scope / Language | Metric | Nemotron 3 Nano Base | Stage 1 Exit | Stage 2 Exit | Stage 3 Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SWE-bench Verified** | 500 clean GitHub issues | Resolved % | **19.8%** | -- | -- | **> 60.0%** |
| **SWE-bench Lite** | 300 Python issues | Resolved % | **17.5%** | **$\ge$ 22.0%** | **$\ge$ 35.0%** | -- |
| **SWE-bench Pro** | 731 multi-file repos | Resolved % | **13.0%** | -- | **> 28.0%** | **> 40.0%** |
| **SWE-bench Multimodal**| 510 UI/DOM frontend tasks| Resolved % | **14.8%** | -- | **> 25.0%** | Pass Gate |
| **HumanEval** | 164 Python functions | Pass@1 (0-shot)| **69.4%** | **$\ge$ 75.0%** | **$\ge$ 82.0%** | **> 86.0%** |
| **LiveCodeBench** | Continuous comp. coding | Pass@1 (Gen) | **26.2%** | -- | **$\ge$ 34.0%** | **> 45.0%** |
| **Tool Calling Compliance**| Multi-turn agent traces | Parse Validity | **95.8%** | **$\ge$ 97.5%** | **$\ge$ 98.5%** | **$\ge$ 99.0%** |

---

## 3. Evaluation Artifacts & Reproducibility

All datasets, evaluation harnesses, and candidate patch diffs are persisted in `./eval_results/`:
- **31B Baseline Data**: [`eval_results/baseline/official_baseline_scores.json`](./eval_results/baseline/official_baseline_scores.json)
- **12B Baseline Data**: [`eval_results/baseline/official_12b_baseline_scores.json`](./eval_results/baseline/official_12b_baseline_scores.json)

---

## 4. Changelog & Milestone Updates

- **2026-09-03**: Established the complete 100% baselines across all 7 benchmarks for both **31B** and **12B** models. Evaluation suites and candidate prediction files verified across 100% of benchmark instances.
