#main/run_02_gan_generator.py

import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)


from src.gan.train_gan import train_gan
from src.gan.generate import generate

def run_gan_generate():
    # ====================
    # Train 
    # ====================
    train_gan()

    # ====================
    # Generate
    # ====================
    generate()

if __name__ == "__main__":
    run_gan_generate()