# Functional Distributional Semantics at Scale

This is the repository for [Functional Distributional Semantics at Scale](https://aclanthology.org/2023.starsem-1.37/) (*SEM 2023).

Functional Distributional Semantics (FDS) attempts to learn truth-conditional meanings of words from distributional information in a corpus. This work extends the applicability of FDS to sentences with more complex structures.

## Getting Started

### Dependencies
* Python = 3.9.15
* PyTorch = 1.10.2
```
conda env create -f environment.yml
```

### Data

* Training the models require linguistically annotated data from [Wikiwoods](https://github.com/delph-in/docs/wiki/WikiWoods):
```
wget http://ltr.uio.no/wikiwoods/1212/
```


### Usage

* See setup.ipynb for the procedure for model training and evaluation.

## Help

Contact Chun Hei Lo at aaronlo326@gmail.com for any assistance!

## Acknowledgments

* [Pytorch Template Project](https://github.com/victoresque/pytorch-template)

