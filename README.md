# AdaMill
AdaMill is a foundational large language model engineered for advanced software intelligence, repository-scale reasoning, and autonomous agent workflows.

## Token Allocation, Loss Masking, & Dataset Sources

| Dataset Component | Volume | Sampling | Epochs | Strategic Role & Loss Masking Rules | Dataset Source Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dolma 3.5 Code Split** | 75.0B | 52.19% | 1.0 | Foundational repository syntax across 14 languages; standard cross-entropy loss. | [AllenAI Dolma 3.5 Pool](https://huggingface.co/datasets/allenai/dolma) |
| **Essential-Web-v1.0** | 40.0B | 27.84% | 1.0 | API docs, PR reviews, and architectural logic; standard cross-entropy loss. | [Essential AI Web Split](https://essential.ai) |
| **Nemotron-Pretraining-Code-v3** | 20.0B | 13.92% | 1.5 | Synthetic diff pairs modeling multi-file downstream class impacts. | [NVIDIA Nemotron Code](https://huggingface.co/nvidia) |
| **Function-Aware FIM Subsplit** | 3.0B | 2.09% | 1.0 | Masked function bodies driven by program-dependence graphs. | [BigCode StarCoder FIM](https://huggingface.co/datasets/bigcode/starcoder) |
| **SERA Agent Workflows** | 2.0B | 1.39% | 3.0 | Multi-turn exploration traces with prompt/tool masking. | [AllenAI Open Coding Agents Trajectories](https://allenai.org) |
| **Scale AI SWE Atlas Tasks** | 1.0B | 0.70% | 2.0 | Refactoring and test suite creation tasks. | [Scale AI SWE-Atlas Repo](https://scale.com) |
| **Algorithmic Diffs** | 1.0B | 0.70% | 2.0 | Numerical debugging examples with stack traces masked and fixes supervised. | [Lean4 Verification Library (fvapps)](https://github.com) |
| **Symbolic Proof & Logic Traces** | 1.0B | 0.70% | 2.0 | Formal verification steps and logic-path supervision. | [Lean4 Mathlib Parsed Proofs](https://github.com/leanprover-community/mathlib4) |
| **DeepSWE Sanitized Split** | 0.5B | 0.35% | 2.5 | Contamination-free execution traces. | [Hugging Face Canonical SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench) |
| **AI2 Tülu Instruction Mix** | 0.5B | 0.35% | 2.0 | Conversational formatting mix for multi-turn prompt compliance. | [AllenAI Tülu 3 SFT Mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| **SWE-bench M Frontend Logic** | 0.4B | 0.28% | 2.0 | DOM manipulation and browser console traces mapped from UI interactions. | [Hugging Face SWE-bench Multimodal](https://huggingface.co/datasets/swe-bench/SWE-bench_Multimodal) |
| **OpenHands Environment Traces** | 0.3B | 0.21% | 4.0 | Terminal interactions, bash return codes, and error-recovery patterns. | [All-Hands AI OpenHands](https://github.com/All-Hands-AI/OpenHands) |
