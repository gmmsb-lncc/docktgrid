.. image:: https://img.shields.io/badge/license-LGPLv3-green
    :target: https://www.gnu.org/licenses/lgpl-3.0.en.html
    :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-black
    :target: https://github.com/psf/black
    :alt: Code style

=========
DockTGrid
=========



Generate voxel representations of protein-ligand complexes for deep learning applications.

.. image:: https://i.imgur.com/VVkQg4t.png
    :align: center
    :width: 50%


    
📌 Features
===========

* GPU-accelerated voxelization of protein-ligand complexes.
* Easy customization of voxel grid channels and parameters.
* Readily usable with `PyTorch <https://pytorch.org/>`_.
* Support for multiple file formats (to be expanded).

  * ✅ PDB
  * ✅ MOL2



🚀 Getting Started
==================

Installation (pip)
------------------
Install DockTGrid using `pip <https://pip.pypa.io/en/stable/>`_::

    $ python -m pip install docktgrid


Development
-----------

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
Python 3.11 is recommended, other versions may work but are not tested.

Clone the repository::

    $ git clone https://github.com/gmmsb-lncc/docktgrid.git
    $ cd docktgrid


Create a new environment using `venv` and activate it::

    $ python3.11 -m venv env
    $ source env/bin/activate

Or if you prefer using `conda <https://docs.conda.io/en/latest/>`_::

    $ conda create --prefix ./venv python=3.11
    $ conda activate ./venv



.. Installation (development)
.. --------------------------

Install the required packages::

    $ python -m pip install -r requirements.txt


.. Testing
.. -------

Run the tests::

    $ python -m pytest tests/



🖥️ Usage
========

See the `documentation <https://docktgrid.readthedocs.io/>`_ for more information on how to use DockTGrid.

There are also some examples in the `notebooks` folder.


📄 License
==========

This project is licensed under the `LGPL v3.0 <https://www.gnu.org/licenses/lgpl-3.0.en.html>`_ license.


📝 Citation
===========

If you use DockTGrid in your research, please cite:

- *da Silva, M. M. P., Guedes, I. A., Custódio, F. L., & Dardenne, L. E. (2024). DockTGrid (0.0.3). Zenodo.* `https://zenodo.org/doi/10.5281/zenodo.10304711`

.. code-block:: bibtex

    @software{mpds2024docktgrid,
        author       = {da Silva, Matheus Müller Pereira and
                  Guedes, Isabella Alvim and
                  Custódio, Fábio Lima and
                  Dardenne, Laurent Emmanuel},
        title        = {DockTGrid},
        month        = mar,
        year         = 2024,
        publisher    = {Zenodo},
        version      = {0.0.3},
        doi          = {10.5281/zenodo.10304711},
        url          = {https://zenodo.org/doi/10.5281/zenodo.10304711}
        }

