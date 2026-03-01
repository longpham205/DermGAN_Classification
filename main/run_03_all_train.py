
# main/run_03_all_train.py

import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from src.classifier.train_baseline import train_baseline
from src.classifier.train_with_basic_aug import train_with_basic_aug
from src.classifier.train_with_gan import train_with_gan

def run_all_train():
    # ====================
    # Train Baseline
    # ====================
    train_baseline()

    # ====================
    # Train Basic AUG
    # ====================
    train_with_basic_aug()

    # ====================
    # Train Gan AUG
    # ====================
    train_with_gan()
    
if __name__ == "__main__":
    run_all_train()

