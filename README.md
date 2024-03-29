# animal-soup
[![Documentation Status](https://readthedocs.org/projects/animal-soup/badge/?version=latest)](https://animal-soup.readthedocs.io/en/latest/?badge=latest) [![CI](https://github.com/hantman-lab/animal-soup/actions/workflows/ci.yml/badge.svg)](https://github.com/hantman-lab/animal-soup/actions/workflows/ci.yml)

The Hantman lab investigates skilled motor control through a head-fixed reach-to-grab task in mice. Over the last few years, a large ground truth dataset has been developed for classifying prominent features of the behavior (`lift`, `handopen`, `grab`, `supinate`, `atmouth`, `chew`). 
![task](https://github.com/hantman-lab/animal-soup/assets/69729525/1aa59e8f-49ea-4d0e-8363-43c483734a95)

The goal of this project is to create a mega-classifier for the Hantman lab to use for future automated behavioral classification.

### Documentation

https://animal-soup.readthedocs.io/

### Installation

For more detailed instructions please see [here](https://animal-soup.readthedocs.io/en/latest/installation.html)

You will need to have [Docker Desktop](https://docs.docker.com/desktop/) installed.

```bash
# clone the repo
git clone https://github.com/hantman-lab/animal-soup.git
cd animal_soup

# build the docker image
docker build -t ansoup .

# run the docker image
docker run --gpus all -w /animal-soup -it --rm -p 8888:8888 -v /home:/home ansoup

# launch jupyter lab from running container on `localhost:9000`
jupyter lab --allow-root --ip=0.0.0.0
```

The `-v /home:/home` assumes that the filesystem you want to mount (where your behavioral data is located) is under 
a directory called `/home`. If your data is located somewhere else you will need to change the mount path when you run the container. 
Mount paths should be in the form `-v /your/local/file/directories:/container/file/structure`.

**Note:** You will only need to build the Docker image once. After you have built the image the first time, you will only need to execute the run command to start the container.

**Important:** A running docker container will not save changes across different runnings of the container. This means that when you stop the docker container instance any changes made to files in the docker environment will not persist when you run the docker container again. However, this **DOES NOT** apply to mounted files. By default your behavior prediction will be saved under your `parent_data_path` that you set before running inference. The `parent_data_path` will be located under the mounted file volume so this will not be an issue. You just need to make sure any jupyter notebooks that you want saved are located under the mounted file volume and **NOT** in the filesystem of the container!

### Data Organization 
Using `pandas.DataFrame` to organize Hantman Lab behavioral data.
![image](https://github.com/hantman-lab/animal-soup/assets/69729525/a3d979f2-9abb-4852-808e-0341b66767cc)

### Behavioral Classification 
Modeled after [DeepEthogram](https://github.com/jbohnslav/deepethogram). 

`animal-soup` has pre-trained models that were initially trained with 1000 ground truth videos and hand-labeled ethograms.
These models were specifically trained for the Hantman Lab reach-to-grab task. 

Please see the demo notebook for downloading the pre-trained models from Zenodo. 

### Interactive Behavior Viewer
Using `pydatagrid` and `fastplotlib` for visualization of behavior with corresponding ethograms.
![viewer](https://github.com/hantman-lab/animal-soup/assets/69729525/55736ffe-303d-4415-b3f1-446c236cc2ba)

### Cleaning Ethograms
Using `fastplotlib` in order to clean ground truth ethograms for model training later.
![cleaner](https://github.com/hantman-lab/animal-soup/assets/69729525/67be413f-a63a-4ee4-8cb7-dd36b3dcaaa9)
  
