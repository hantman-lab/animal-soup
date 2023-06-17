# animal-soup
The Hantman Lab investigates skilled motor control through a head-fixed reach-to-grab task in mice. Over the last few years, a large ground truth dataset has been developed for classifying prominent features of the behavior (`lift`, `handopen`, `grab`, `supinate`, `atmouth`, `chew`). 

The goal of this project is to create a mega-classifier for the Hantman lab to use for future automated behavioral classification.

### Data Organization 
Using `pandas.DataFrame` to organize Hantman Lab behavioral data.
![image](https://github.com/hantman-lab/animal-soup/assets/69729525/177c2af0-4b6a-4202-ab3f-1f18fc8df649)

### Interactive Behavior Viewer
Using `pydatagrid` and `fastplotlib` for visualization of behavior with corresponding ethograms.
![behavior_viewer](https://github.com/hantman-lab/animal-soup/assets/69729525/235bf67e-88e0-4d8a-a34c-4848b48d94f8)

### Fixing Ethograms
Using `fastplotlib` in order to clean ground truth ethograms for model training later.
![image](https://github.com/hantman-lab/animal-soup/assets/69729525/a6d1d37c-5952-48b1-8fc9-9997d3346564)

### Future Developments
- Improvements to the UI (keyboard events to clean ethograms as opposed to buttons)
- CNN for behavior annotation prediction
  
