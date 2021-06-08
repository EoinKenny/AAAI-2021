# On Generating Plausible Counterfactual and Semi-Factual Explanations for Deep Learning


This repository is the official implementation "On Generating Plausible Counterfactual and Semi-Factual Explanations for Deep Learning" (https://ojs.aaai.org/index.php/AAAI/article/view/17377). 

![alt text](https://github.com/EoinKenny/AAAI-2021/blob/master/imgs/overview.png)



## Requirements

To install requirements:

```setup
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```


## Evaluation

To run the counterfactual experiment in the main paper, run:

```eval
python run_counterfactual_expt.py
```

CEM and Proto-CF do not run (I commented them out of the python file) as there is some error in the Alibi installation which has cropped up in the year since I ran the initial experiments, unfortunately this is beyond my control to fix.

## Pre-trained Models

You can download pretrained files needed here:

- https://drive.google.com/drive/folders/1GfL4dlztWrxjZoJhShlBJW8zdVS1iT3D?usp=sharing

Put 'distribution data' in the data folder. Put 'pred_features.pickle' in the data folder. Put 'generator.pth' in the weights folder.

## Results

Table of results from counterfactual experiment:

![alt text](https://github.com/EoinKenny/AAAI-2021/blob/master/imgs/results.png)


## Cite Bibtext

@article{Kenny_Keane_2021, title={On Generating Plausible Counterfactual and Semi-Factual Explanations for Deep Learning}, 
volume={35}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/17377}, number={13}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Kenny, Eoin M. and Keane, Mark T}, year={2021}, 
month={May}, 
pages={11575-11585}
 }