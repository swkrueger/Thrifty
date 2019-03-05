Thrifty
=======

Thrifty is proof-of-concept SDR software for TDOA positioning using inexpensive
SDR hardware such as the RTL-SDR.

Requirements
------------
 - `Python <http://www.python.org/>`_ 2.7 or later
 - `Numpy <http://www.numpy.org/>`_
 - [Optional] `matplotlib <http://matplotlib.org/>`_

Installation
------------
To install thrifty::

    $ sudo apt-get install python-pip
    $ sudo pip install .

or::

    $ sudo python setup.py install

In addition to installing thrifty into the system's Python environment, this
will also download and install the Python module requirements from `PyPI
<http://pypi.python.org/>`_.

To install thrifty in developer mode, which creates a symbolic link from the
source location to the user's install location::

    $ make dev

Thrifty requires ``fastcard`` to capture data. Refer to
`fastcard/README.md <fastcard/README.md>`_ for installation instructions.
Furthermore, refer to `fastdet/README.md <fastdet/README.md>`_ for more
information regarding ``fastdet``, a fast replacement for ``thrifty detect``.

Refer to `rpi/installation.md <rpi/installation.md>`_ for instructions on
configuring an Raspberry Pi 3 for use as an inexpensive TDOA receiver.

Usage
-----
A command-line interface (CLI) is available through the ``thrifty`` command.
Run ``thrifty help`` for a summary of the modules that are available through
the CLI.

Typical CLI workflow::

    $ cd example/
    $ vim detector.cfg   # edit config

    $ # On RX0:
    $ thrifty capture rx0.card
    $ thrifty detect rx0.card -o rx0.toad

    $ # On RX1:
    $ thrifty capture rx1.card
    $ thrifty detect rx0.card -o rx1.toad

    $ # On server:
    $ thrifty identify rx0.toad rx1.toad
    $ thrifty match
    $ thrifty tdoa
    $ thrifty pos


Alternatively, use the Makefile::

    cd example/
    vim detector.cfg   # edit config
    thrifty capture cards/rxX.card
    make


Detection on slow hardware: see `fastcard <fastcard/README.md>`_ and
`fastdet <fastdet/README.md>`_.


Cookbook:

 - Live detection without capturing (for monitoring)::

       thrifty capture - 2>/dev/null | thrifty detect - -o /dev/null

 - Parallel capture-and-detection::

       thrifty capture rx.card
       tail -f rx.card | thrifty detect -


For advanced use cases, use the thrifty API from Python or IPython, e.g.:

.. code-block:: python

    """Plot histogram of SoA offsets for all detections from TX #0."""
    
    import matplotlib.pyplot as plt
    
    from thrifty import toads_data
    
    toads = toads_data.load_toads(open('rx.toads', 'r'))
    data = toads_data.toads_array(toads, with_ids=True)
    tx0_data = data[data['txid'] == 0]
    plt.hist(tx0_data['offset'], bins=20)
    plt.show()

Publications
------------
Thrifty forms part of the dissertation at https://hdl.handle.net/10394/25449. Please cite this dissertation when using Thrifty in your work:

.. code-block:: bibtex

    @mastersthesis{kruger2016inexpensive,
      title={An inexpensive hyperbolic positioning system for tracking wildlife using off-the-shelf hardware},
      author={Kr{\"u}ger, Schalk Willem},
      year={2016},
      school={North-West University (South Africa), Potchefstroom Campus}
    }

Refer to https://swk.za.net/publications for contact information.
