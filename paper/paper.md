---
title: "DockTGrid: A Python-Based Tool for Generating Deep Learning-Ready Voxel Grids of Molecular Complexes"

tags:
  - scoring function
  - molecular docking
  - binding affinity prediction
  - structure-based drug design
  - virtual screening
  - deep learning
  - python

authors:
  - name: Matheus Müller Pereira da Silva
    orcid: 0000-0002-0659-6365
    affiliation: 1

  - name: Isabella Alvim Guedes
    orcid: 0000-0002-9761-4804
    affiliation: 1

  - name: Fábio Lima Custódio
    orcid: 0000-0001-5134-2340
    affiliation: 1

  - name: Laurent Emmanuel Dardenne
    orcid: 000-0001-8518-8062
    corresponding: true
    affiliation: 1

affiliations:
 - name: Grupo de Modelagem Molecular em Sistemas Biológicos (GMMSB), Laboratório Nacional de Computação Científica, Ministério da Ciência, Tecnologia e Inovação (LNCC/MCTI), Petrópolis - RJ, Brazil
   index: 1

date: 21 March 2024
bibliography: paper.bib
---

# Summary
Deep learning (DL) techniques are at the forefront of advancements 
in molecular modelling tools, significantly impacting the 
field of computer-aided drug design (CADD) [@chen2018rise; @zhang2017machine; @shen2020machine; @guedes2018empirical].
In this context, voxel grid representations have emerged as a 
straigthforward way to depict the three-dimensional molecular structures of protein-ligand complexes.
These representations are used as input to DL models, such as convolutional
neural networks (CNNs), to make inferences on a range of CADD tasks, including molecular
docking [@mcnutt2021gnina; @ragoza2017protein], 
binding affinity prediction [@ragoza2017protein; @jimenez2018k; @stepniewska2018development; @liu2021octsurf; @li2019deepatom], 
binding site prediction [@liang2022efficient; @jimenez2017deepsite], molecular generation [@ragoza2022generating] and
virtual screening [@ragoza2017protein; @sunseri2021virtual]. In this work, we introduce a Python-based software, `DockTGrid`,
which provides a set of tools to generate, customize and manipulate voxel grid representations
of protein-ligand complexes ready to be used as input to DL models. 

Voxel grids approach the challenge of molecular representation through a 
computer vision lens. A voxel (_volume element_) corresponds to a value on a regular grid in the
three-dimensional space. A voxel grid is a 4-dimensional tensor with dimensions $c \times w \times h \times d$,
where $w$, $h$, and $d$ are the grid's width, height, and depth, respectively, and $c$ is the number of channels
(similar to RGB channels in 2D image representations).  Voxel grids effectively capture 
the 3D spatial relationships between atoms and their local environment, making them
ideal inputs for CNNs equipped 3D convolutional layers.

The values each voxel holds are defined by the occupancy model used.
Space-filling occupancy models, which lead to denser representations and 
thereby reduce unutilized space within the grid, are frequently employed. 
These models incorporate the van der Waals radius of atoms and employ 
a pair correlation function to depict the electronic surface in space. 
`DockTGrid` includes a space-filling occupancy model [@jimenez2017deepsite], which is 
mathematically expressed as:

$$v(d) = 1 - \exp(- (\frac{r_{vdw}}{d})^{12}).$$ 

Here, $v(d)$ represents the voxel occupancy value at a distance $d$ 
from the atom's center, with $r_{vdw}$ denoting the van der Waals 
radius of the atom.

In CADD and molecular contexts, 
grid channels can represent various physicochemical properties of atoms. 
For instance, channels may be designated to reflect specific atomic 
properties such as hydrophobicity, hydrogen bond acceptance or donation, 
ionizability (either positive or negative), partial atomic charges, 
and the presence of metals [@jimenez2018k; @stepniewska2018development]. Alternatively, 
channels might  represent different chemical elements, including carbon, hydrogen, 
oxygen, and nitrogen [@ragoza2022generating; @li2019deepatom]. \autoref{fig:channels} demonstrates how channels 
for carbon and hydrogen atoms in a ligand are depicted using the 
space-filling occupancy model previously described.

![Voxel-Based Representation of Carbon and Hydrogen Atoms Generated with `DockTGrid`. In this visualization, 
carbon (a) and hydrogen (b) atoms within the adenosine 2'-monophosphate molecule 
(PDB ID: 6RNT) are depicted using a space-filling occupancy model across two 
distinct channels. \label{fig:channels}](./fig.jpg)


# Statement of need
Related software packages — available on public package repositories (e.g., [PyPI](https://pypi.org/), for Python) — 
offer capabilities for generating voxel grid representations of molecular data tailored for 
DL applications. 
Notable examples include [`libmolgrid`](https://github.com/gnina/libmolgrid) [@sunseri2020libmolgrid], 
[`moleculekit`](https://github.com/Acellera/moleculekit) [@doerr2016htmd], 
and [`deepchem`](https://github.com/deepchem/deepchem) [@ramsundar2019deep]. However, 
users may encounter challenges with these packages in feature customization or programming language diversity. 
In some of these, not all functionalities are developed in Python; incorporating programming languages like C++, 
which could limit accessibility for users primarily familiar with Python in the 
DL community.


`DockTGrid` was developed entirely in Python 3 and is built atop the widely-used DL 
library [`PyTorch`](https://pytorch.org/). This foundation enables seamless integration and rapid, GPU-accelerated 
grid generation. The software offers customizable grid channels through a straightforward 
and user-friendly object-oriented interface, as illustrated in the software documentation 
(available at [https://docktgrid.readthedocs.io](https://docktgrid.readthedocs.io)). Designed for 
use by researchers and DL practitioners in the CADD
field, `DockTGrid` presents an accessible and open-source solution for generating voxel grid 
representations of protein-ligand complexes.


# Acknowledgements
This work was supported by the Brazilian agencies Conselho Nacional de Desenvolvimento
Científico e Tecnológico (CNPq) (grant number 309744/2022-9); and the Fundação Carlos
Chagas Filho de Apoio à Ciência (FAPERJ) (grant numbers E-26/010.001415/2019,
E-26/211.357/2021, E-26/ 200.393/2023, E-26/200.608/2022, E-26/210.372/2022).
We gratefully acknowledge the support of the Brazilian Sistema Nacional de Processamento
de Alto Desempenho (SINAPAD) and the availability of the computational resources provided
by the Supercomputer SDumont (LNCC/MCTI)

# Conflict of Interest
The authors declare that the research was conducted in the absence of any commercial or
financial relationships that could be construed as a potential conflict of interest.

# References