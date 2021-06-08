# On Generating Plausible Counterfactual and Semi-Factual Explanations for Deep Learning


This repository is the official implementation "On Generating Plausible Counterfactual and Semi-Factual Explanations for Deep Learning" (https://ojs.aaai.org/index.php/AAAI/article/view/17377). 

![alt text](https://github.com/EoinKenny/AAAI-2021/blob/master/imgs/overview.png)



## Requirements

To install requirements:

```setup
python3 -m vena mien
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


## Cite

@article{Kenny_Keane_2021, title={On Generating Plausible Counterfactual and Semi-Factual Explanations for Deep Learning}, volume={35}, url={https://ojs.aaai.org/index.php/AAAI/article/view/17377}, abstractNote={There is a growing concern that the recent progress made in AI, especially regarding the predictive competence of deep learning models, will be undermined by a failure to properly explain their operation and outputs. In response to this disquiet, counterfactual explanations have become very popular in eXplainable AI (XAI) due to their asserted computational, psychological, and legal benefits. In contrast however, semi-factuals (which appear to be equally useful) have surprisingly received no attention. Most counterfactual methods address tabular rather than image data, partly because the non-discrete nature of images makes good counterfactuals difficult to define; indeed, generating plausible counterfactual images which lie on the data manifold is also problematic. This paper advances a novel method for generating plausible counterfactuals and semi-factuals for black-box CNN classifiers doing computer vision. The present method, called PlausIble Exceptionality-based Contrastive Explanations (PIECE), modifies all “exceptional” features in a test image to be “normal” from the perspective of the counterfactual class, to generate plausible counterfactual images. Two controlled experiments compare this method to others in the literature, showing that PIECE generates highly plausible counterfactuals (and the best semi-factuals) on several benchmark measures.}, number={13}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Kenny, Eoin M. and Keane, Mark T}, year={2021}, month={May}, pages={11575-11585} }