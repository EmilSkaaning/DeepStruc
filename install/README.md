[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6221f17357a9d20c9a729ecb)  |  [Paper](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00086e)

# Install
To run DeepStruc you will need some packages, which are dependent on your computer specifications. 
This includes [PyTorch](https://pytorch.org/) and 
[PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), make sure that
the versions match and [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) is version 1.7.2.
All other packages can be install through the requirement file. 
First go to your desired conda environment.
 ```
conda activate <env_name>
``` 
Or create one and activate it afterwards.
```
conda create --name env_name python=3.7
``` 
Now install the required packages through the requirement files. Install [DiffPy-CMI](https://www.diffpy.org/products/diffpycmi/index.html) (see how to [HERE](https://www.diffpy.org/products/diffpycmi/index.html))
if you want to be able to use the gen_data script.
```
pip install -r requirements.txt
``` 


