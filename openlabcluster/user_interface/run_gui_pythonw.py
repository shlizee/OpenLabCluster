"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Developed in UW NeuroAI Lab by Moishe Keselman.
"""
import os
import subprocess

# Launches the OpenLabCluster as a subprocess.
def run():
    subprocess.Popen(['pythonw', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_gui.py')]).wait()

if __name__ == '__main__':
    run()
