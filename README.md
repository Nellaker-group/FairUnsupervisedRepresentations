# Adjusting for Confounding in Unsupervised Latent Representations of Images

This repository collects implementation details for the analyses reported in the paper [Adjusting for Confounding in Unsupervised Latent Representations of Images.](url)

# Contents

- [Dependencies](#dependencies)
- [Data](#data)
- [Background and Motivations](#background-and-motivations)
- [Training](#training)
- [Models](#models)
- [References](#references)

# Dependencies

`python 3.5.2`
`skimage 0.13.1`

`numpy 1.14.3`
`torch 0.4.0`

# Data 

We used the BBBC021v1 image set, which is a resource freely available to download through the Broad bioimage benchmark collection web server ([https://data.broadinstitute.org/bbbc/](https://data.broadinstitute.org/bbbc/)). These data capture phenotype changes of cancer cell lines exposed to a compendium of drugs. The imaging set was annotated as benchmark data to develop phenotypic profiling methods and, in particular, to validate their ability to predict the molecular mechanism of action (MOA) for a collection of compounds. Phenotypes were captured, across 10 weeks (batches), by labelling cells with DAPI, Tubulin, and Actin, thereby generating a triplet of single channel images (one per fluorescent marker) for each treatment.

For each image in the BBBC021v1 set, we detected cell nuclei using the algorithm difference of Gaussians on the DAPI channel. We cropped patches of <img src="https://latex.codecogs.com/svg.latex?\inline&space;128&space;\times&space;128" title="128 \times 128" /> pixels centred around each nucleus, and annotated <img src="https://latex.codecogs.com/svg.latex?\inline&space;128&space;\times&space;128&space;\times&space;3" title="128 \times 128 \times 3" /> images by concatenating patches from the DAPI, Tubulin, and Actin channels. Representative training images are displayed in Figure 1.

![](examples.png)

# Background and Motivations

Previous works using BBBC021v1 data have shown it’s possible to learn deep representations capturing biological (MOA) knowledge [[1,2,3]](#references). Furthermore, one study has shown the presence of a strong confounder, as BBBC021v1 representations also encode spurious knowledge capable of discriminating treatments according to their imaging batch [[2]](#references).  However, no detailed analyses have been undertaken to asses the impact such bias is having on learned data representations, in particular whether it’s feasible to remove nuisance knowledge encoded in learned embeddings. In our work, we showed how it’s possible to learn unbiased representations encoding biological (MOA) knowledge and invariant to the batch effect confounder.

# Training

<img src="https://latex.codecogs.com/svg.latex?\textrm{min}_{\,\theta}\,\,\,\mathcal{L}_{\textrm{CAE}}(\theta)-\lambda\,\mathcal{L}_{\textrm{adv}}(\mathbold{\theta},&space;w)" title="\textrm{min}_{\,\theta}\,\,\,\mathcal{L}_{\textrm{CAE}}(\theta)-\lambda\,\mathcal{L}_{\textrm{adv}}(\mathbold{\theta}, w)" />

<img src="https://latex.codecogs.com/gif.latex?\textrm{min}_{\,w}\,\,\mathcal{L}_{\textrm{adv}}(\theta,&space;w)" 

# Models

# References

1. Nick Pawlowski et al., [“Automating morphological profiling with generic deep convolutional
networks”,](https://www.biorxiv.org/content/early/2016/11/02/085118) bioRxiv 2016.

2. D. Michael Ando et al., [“Improving Phenotypic Measurements in
High-Content Imaging Screens”,](https://www.biorxiv.org/content/early/2017/07/10/161422) bioRxiv 2017.

3. Alexander Kensert et al., [“Transfer learning with deep convolutional neural network for classifying cellular morphological changes”,](https://www.biorxiv.org/content/early/2018/06/14/345728) bioRxiv 2018.


