#!/usr/bin/env python3
"""
LiveCodeBench (LCB) & HumanEval+ Benchmark Runner for Lovelace SWE Agent.
Evaluates code generation, reasoning, bug repair, and test generation on contamination-free competitive coding tasks.

Usage:
    # Run LiveCodeBench (LCB) on Code Generation
    python run_eval_livecodebench.py --task code_generation --model gemma4_31b

    # Run on Self-Repair / Code Execution tasks
    python run_eval_livecodebench.py --task code_execution --model gemma4_31b

    # Run on HumanEval / HumanEval+
    python run_eval_livecodebench.py --task humaneval --model gemma4_31b
"""

import argparse
import os
import sys
import time

TASKS = {
    "code_generation": "LiveCodeBench: Problem solving and generation (Post-cutoff LeetCode/AtCoder)",
    "code_execution": "LiveCodeBench: Trace execution and state output prediction",
    "test_generation": "LiveCodeBench: Synthesizing adversarial test cases",
    "humaneval": "HumanEval / HumanEval+: Python function synthesis Pass@1 / Pass@10",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run LiveCodeBench & HumanEval evaluation.")
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="code_generation",
        choices=list(TASKS.keys()),
        help="Target benchmark task (default: code_generation)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemma4_31b",
        help="Model checkpoint path or config name (default: gemma4_31b)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--n_samples",
        "-n",
        type=int,
        default=1,
        help="Number of samples per instance for Pass@k (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./eval_results/livecodebench",
        help="Directory to save evaluation results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=================================================================")
    print("  Lovelace LiveCodeBench (LCB) & Algorithmic Evaluation")
    print("=================================================================")
    print(f"Evaluation Task  : {args.task}")
    print(f"Task Description : {TASKS[args.task]}")
    print(f"Model Under Test : {args.model}")
    print(f"Sampling Temp    : {args.temperature}")
    print(f"Samples / Item   : {args.n_samples}")
    print(f"Output Directory : {args.output_dir}")
    print("=================================================================\n")

    os.makedirs(args.output_dir, exist_ok=True)
    print("Harness configured. Evaluation ready to execute on MI355X cluster.")


if __name__ == "__main__":
    main()
