# animal-soup
[![Documentation Status](https://readthedocs.org/projects/animal-soup/badge/?version=latest)](https://animal-soup.readthedocs.io/en/latest/?badge=latest) [![CI](https://github.com/hantman-lab/animal-soup/actions/workflows/ci.yml/badge.svg)](https://github.com/hantman-lab/animal-soup/actions/workflows/ci.yml)

The Hantman lab investigates skilled motor control through a head-fixed reach-to-grab task in mice. Over the last few years, a large ground truth dataset has been developed for classifying prominent features of the behavior (`lift`, `handopen`, `grab`, `supinate`, `atmouth`, `chew`). 
![task](https://github.com/hantman-lab/animal-soup/assets/69729525/1aa59e8f-49ea-4d0e-8363-43c483734a95)

The goal of this project is to create a mega-classifier for the Hantman lab to use for future automated behavioral classification.

### Documentation

https://animal-soup.readthedocs.io/

### Installation

```bash
git clone https://github.com/hantman-lab/animal-soup.git
cd animal_soup

pip install -e .
```

### Data Organization 
Using `pandas.DataFrame` to organize Hantman Lab behavioral data.
![image](https://github.com/hantman-lab/animal-soup/assets/69729525/a3d979f2-9abb-4852-808e-0341b66767cc)

### Behavioral Classification 
Modeled after [DeepEthogram](https://github.com/jbohnslav/deepethogram). 

`animal-soup` has pre-trained models that were initially trained with 1000 ground truth videos and hand-labeled ethograms.
These models were specifically trained for the Hantman Lab reach-to-grab task. 

### Interactive Behavior Viewer
Using `pydatagrid` and `fastplotlib` for visualization of behavior with corresponding ethograms.
![viewer](https://github.com/hantman-lab/animal-soup/assets/69729525/55736ffe-303d-4415-b3f1-446c236cc2ba)

### Cleaning Ethograms
Using `fastplotlib` in order to clean ground truth ethograms for model training later.
![cleaner](https://github.com/hantman-lab/animal-soup/assets/69729525/67be413f-a63a-4ee4-8cb7-dd36b3dcaaa9)
  
