# Data introduction
This folder contains __graphs__-, __utils__ folder and a gen_data.py script.  
The __graphs__ folder contains a small set of graph which can be used to varify that DeepStruc is running. To
generate a broader distribution of monometallic nanoparticles (MMNPs) for training, validation and testing the gen_data.py 
script is needed.
  

1. [Generate data](#generate-data)
    1. [Generate data arguments](#generate-data-arguments)
2. [Generel data structure](#generel-data-structure)
    1. [Monometallic nanoparticles](#monometallic-nanoparticles)
    2. [Graphs and PDFs](#graphs-and-pdfs)

# Generate data
To generate more data run the gen_data.py script. The scripts takes a range of arguments which are all descriped below
or use the __help__ command to produce the parameter list. The __help__ argument will also show default values.  
```
python gen_data.py --help
>>> usage: gen_data.py [-h] [-d DIRECTORY] [-a ATOMS [ATOMS ...]]
>>>                    [-t {SC,FCC,BCC,HCP,Ico,Dec,Oct} [{SC,FCC,BCC,HCP,Ico,Dec,Oct} ...]] 
>>>                    [-n NUM_ATOMS] [-i INTERPOLATION] [-q QMIN] [-Q QMAX]  
>>>                    [-r RMIN] [-R RMAX] [-rs RSTEP] [-b BISO]    
>>>
>>> Generating structures, graphs and conditional PDFs for DeepStruc.    
>>> ...
```

## Generate data arguments
List of possible arguments.  
 
| Arg | Description | Example |  
| --- | --- |  --- |  
| `-h` or `--help` | Prints help message. |    
| `-d` or `--directory` | Prints help message. __str__ | `-d new_data`  |   
| `-a` or `--atoms` | An atom or list of atoms. __str__| `-a Nb W Mo`  |
| `-t` or `--structure_type` | A single or list of structure types. Possible structure types are: SC, FCC, BCC, HCP, Ico, Dec and Oct __str__| `-t SC Ico`  |  
| `-n` or `--num_atoms` | Maximum number of possible atoms in structures generated. __int__| `-n 200`  |  
| `-i` or `--interpolation` | Prints help message. __int__| `-i 3`  |  
| `-q` or `--qmin` | Smallest scattering amplitude for simulated PDFs. __float__| `-q 0.2`  |  
| `-Q` or `--qmax` | Largest scattering amplitude for simulated PDFs. __float__| `-Q 22.3`  |  
| `-p` or `--qdamp` | PDF Gaussian dampening factor due to limited Q-resolution. Not applied when equal to zero.. __float__| `-p 22.3`  |  
| `-r` or `--rmin` | Smallest r-value for simulated PDFs. __float__| `-r 1.5`  |  
| `-R` or `--rmax` | Largest r-value for simulated PDFs. __float__| `-R 20.0`  |  
| `-s` or `--rstep` | r-grid spacing for simulated PDFs. __float__| `-s 0.1`  |  
| `-e` or `--delta2` | Coefficient for (1/r**2) contribution to the peak sharpening.. __float__| `-e 3.5`  |  
| `-b` or `--biso` | Isotropic Atomic Displacement Parameter for simulated PDFs. __float__| `-b 0.2`  |  

  
# Generel data structure

## Monometallic nanoparticles

## Graphs and PDFs
![alt text](../img/graph_rep.png "Graphs representation of MMNPs.")
 
