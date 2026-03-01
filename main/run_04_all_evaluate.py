# main/run_04_all_evaluate.py

import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from src.classifier.evaluate import evaluate

def run_all_evaluate():
    # ====================
    # Train Baseline
    # ====================
    evaluate('exp_01_baseline')

    # ====================
    # Train Basic AUG
    # ====================
    evaluate('exp_02_base_aug')

    # ====================
    # Train Gan AUG
    # ====================
    evaluate('exp_03_gan_aug')

if __name__ == "__main__":
    run_all_evaluate()