
# On-Generating-Plausible-Counterfactual-and-Semi-Factual-Explanations-For-Deep-Learning-Supplementary-Code

![alt text](https://github.com/EoinKenny/AAAI-2021/blob/master/imgs/overview.png)


This repo is for our paper "On Generating Plausilbe Counterfactual and Semi-Factual Explanations for Deep Learning" at AAAI-2021. The paper link is here

??????

Unfortunately, when I ran the experiments (almost a year ago) my original environment has become lost. I have tried to re-create it, but it seems that in the year past dependencies beyond my control have made it impossible to accurately recreate the results in a single script with a single environment.

However, I have included all the code here to make it easy to reproduce some results. Please run the following commands to have a compatable environment.


Python3 -m venv myenv
source myenv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name= myenv

This will setup an environment which will run with jupyter notebooks and import the environment to the notebook's available kernels.


Requirements
Pip install alibi (version '0.5.5')
pip install torch==1.6.0 torchvision==0.7.0

It will be possible to run the 'run_counterfactual_expt.py' file with this environment, but it will crash at using CEM/Proto-CF in the alibi libraries, if you wish, you could remove these from the script and just collect the data for our method PIECE and the two others (Min-Edit, and C-Min-Edit). This would be enough to recreate most of the paper. In addition, perhaps you create a different environment to collect the data for CEM/Proto-CF.

Since I cannot recreate the full experiments, I have only included the files necessary to generate explanations for the incorrect classifications on MNIST, to simlify the repo.

==================================================
Jupyter Notebook
==================================================

The notebook above 'Example Explanation on MNIST.ipynb' is probabaly what you want to focus on. I provide step-by-step instructions on how to generate explanations with PIECE for MNIST, along with an example and all latent inputs for the GAN saved which saves you a lot of time.

# AAAI-2021
# AAAI-2021
