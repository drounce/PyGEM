# Introduction
The Python Glacier Evolution Model (PyGEM) is an open-source glacier evolution model coded in Python that is designed to model the transient evolution of glaciers on regional and global scales. Each glacier is modeled independently using a given time step and elevation bins. The model computes the climatic mass balance (i.e., snow accumulation minus melt plus refreezing) for each elevation bin and each monthly time step. Glacier geometry is updated annually. The model outputs a variety of data including monthly mass balance and its components (accumulation, melt, refreezing, frontal ablation, glacier runoff),  and annual volume, volume below sea level, and area.

PyGEM has a modular framework that allows different schemes to be used for model calibration or model physics (e.g., ablation, accumulation, refreezing, glacier dynamics). The most recent version of PyGEM, published in <em>Science</em> [(Rounce et al. 2023)](https://www.science.org/doi/10.1126/science.abo1324), has been made compatible with the Open Global Glacier Model [(OGGM)](https://oggm.org/) to both leverage the pre-processing tools (e.g., digital elevation models, glacier characteristics) and their advances with respect to modeling glacier dynamics and ice thickness inversions.

```{note}
Looking for a quick overview? Check out the [Model Structure and Workflow](model_structure_and_workflow_target).
<br>Want to read some studies?  Check out our [publications](publications_target)!
<br>Want to see what PyGEM can do? Check out this [presentation about PyGEM's latest developments](https://www.youtube.com/watch?v=gaGzEIjIJlc)!
```

## Citing PyGEM
The most recent version of PyGEM was published in <em>[Science](https://www.science.org/doi/10.1126/science.abo1324)</em>. Therefore, if using PyGEM, please cite:
<br><br>Rounce, D.R., Hock, R., Maussion, F., Hugonnet, R., Kochtitzky, W., Huss, M., Berthier, E., Brinkerhoff, D., Compagno, L., Copland, L., Farinotti, D., Menounos, B., and McNabb, R.W. (2023). “Global glacier change in the 21st century: Every increase in temperature matters”, <em>Science</em>, 379(6627), pp. 78-83, doi:10.1126/science.abo1324.

```{figure} _static/science_cover.jpg
---
width: 50%
---
```

## Contact us
The PyGEM community is growing! As the community grows, we hope individuals will continue to support one another. For now, if you have questions, we recommend emailing David Rounce (drounce@cmu.edu).