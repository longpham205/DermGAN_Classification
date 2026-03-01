import os, sys
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_dir)

from main.run_01_preprocess     import run_preprocess
from main.run_02_gan_generate   import run_gan_generate
from main.run_03_all_train      import run_all_train
from main.run_04_all_evaluate   import run_all_evaluate
from main.run_05_analysis       import run_analysis

if __name__ == "__main__":
    # ====================
    # 01. PREPROCESSING
    # ====================
    run_preprocess()
    
    # ====================
    # 02. GAN GENERATOR 
    # ====================
    run_gan_generate()
    
    # ====================
    # 03. ALL TRAIN
    # ====================
    run_all_train()
    
    # ====================
    # 04. ALL EVALUATE
    # ====================
    run_all_evaluate()
    
    # ====================
    # 05. ANALYSIS
    # ====================
    run_analysis()
    

    