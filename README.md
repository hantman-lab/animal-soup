# animal-soup
The Hantman lab investigates skilled motor control through a head-fixed reach-to-grab task in mice. Over the last few years, a large ground truth dataset has been developed for classifying prominent features of the behavior (`lift`, `handopen`, `grab`, `supinate`, `atmouth`, `chew`). 

The goal of this project is to create a mega-classifier for the Hantman lab to use for future automated behavioral classification.

### Data Organization 
Using `pandas.DataFrame` to organize Hantman Lab behavioral data.
![image](https://github.com/hantman-lab/animal-soup/assets/69729525/e99c6a04-02a4-4584-b712-9d052115a877)

### Interactive Behavior Viewer
Using `pydatagrid` and `fastplotlib` for visualization of behavior with corresponding ethograms.
![behavior_viewer](https://github.com/hantman-lab/animal-soup/assets/69729525/235bf67e-88e0-4d8a-a34c-4848b48d94f8)

### Fixing Ethograms
Using `fastplotlib` in order to clean ground truth ethograms for model training later.
![image](https://github.com/hantman-lab/animal-soup/assets/69729525/38154dd3-ee5f-4591-8564-a6f450ddfd53)


### Future Developments
- CNN for behavior annotation prediction
  
