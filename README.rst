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

Thrifty requires ``fastcard`` to capture data. Refer to ``fastcard/README.md`` for
installation instructions.

Usage
-----
A command-line interface (CLI) is available through the ``thrifty`` command. Run
``thrifty help`` for a summary of the modules that are available through the CLI.

Typical CLI workflow::

    $ cd example/
    $ vim thrifty.cfg   # edit config

    $ # On RX0:
    $ thrifty capture -o rx0.card
    $ thrifty detect rx0.card -o rx0.toad

    $ # On RX1:
    $ thrifty capture -o rx1.card
    $ thrifty detect rx0.card -o rx1.toad

    $ # On server:
    $ thrifty integrate *.toad -o rx.toads
    $ thrifty match rx.toads -o rx.match
    $ thrifty clock_sync rx.toads rx.match


Alternatively, use the Makefile::

    cd example/
    vim thrifty.cfg   # edit config
    thrifty capture -o cards/rxX.card
    make


Cookbook:

 - Live detection without capturing (for monitoring)::

       thrifty capture - 2>/dev/null | thrifty detect - -o /dev/null

 - Parallel capture-and-detection::

       thrifty capture rx.card
       tail -f rx.card | thrifty detect -


For advanced use cases, use the thrifty API from Python or IPython, e.g.:

.. highlight:: python

    """Plot histogram of SoA offsets for all detections from TX #0."""
    
    import matplotlib.pyplot as plt
    
    from thrifty import toads_data
    
    toads = toads_data.load_toads(open('rx.toads', 'r'))
    data = toads_data.toads_array(toads, with_ids=True)
    tx0_data = data[data['txid'] == 0]
    plt.hist(tx0_data['offset'], bins=20)
    plt.show()
