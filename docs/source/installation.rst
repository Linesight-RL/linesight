============
Installation
============

Prerequisites
-------------

Linesight requires:
    - Python >=3.10
    - PyTorch >=2.0 (check the `official website <https://pytorch.org/get-started/locally/>`_ for specific instructions)
    - a Nvidia GPU with CUDA
    - 16 GB RAM
    - `Trackmania Nations Forever <https://store.steampowered.com/app/11020/TrackMania_Nations_Forever/>`_ patched with `TMInterface 2.1.4 <https://www.donadigo.com/tminterface/>`_.

This project is compatible with Windows and Linux.

Python project setup
--------------------

Clone the repository and install the project. We recommend using virtual environments (conda, mamba, pipenv, ...).

Conda / Mamba
~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/pb4git/linesight && cd linesight
    conda install --file requirements_conda.txt
    pip install -e .

pip
~~~

.. code-block:: bash

    git clone https://github.com/pb4git/linesight && cd linesight
    pip install -e .


Platform-specific instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is required to provide a way for the project to launch multiple instances of the game. The instructions to do so depend on the operating system and are detailed below.

Windows
.......

In the folder ``/scripts/``, create a shortcut called ``TMInterface.lnk`` pointing to the TMInterface executable. On a standard system with a default Steam TMNF installation, the shortcut points to ``C:\Program Files (x86)\Steam\steamapps\common\Trackmania Nations Forever\TMInterfaceTesting.exe``.

Linux
.....
In the folder ``/scripts/``, create a bash script that takes an integer ``port`` as its single argument. The script should start the game configured to listen on port ``port`` for TMInterface communications. The scripts ``/scripts/launch_game_pb.sh`` and ``/scripts/launch_game_agade.sh`` are working examples on the authors' systems.

.. note::
    The authors have experienced improved FPS when executing TMNF within wine with the following setup:

    .. code-block:: bash

        sudo apt install winetricks
        winetricks dxvk

    and launching the game with ``exec gamemoderun wine (...)`` (see ``/scripts/launch_game_pb.sh`` for example).

Game configuration
------------------

The game must be configured (via ``TmForeverLauncher.exe`` in the game's installation directory) to run in windowed mode.
We recommend adjusting game settings to run at the lowest resolution available with low graphics quality.

.. note::
   There is a compromise to be made between *training speed* which increases with FPS and *trained performance* which increases with image quality. Users can experiment with their setup, or use the authors' configuration file `available here <https://link_to_file.com>`_.

User config
-----------

Open the file ``config_files/user_config.py`` and make modifications relevant to your system:

.. code-block:: python

    username = "username"  # Username of the TMNF account

    # Path where Python_Link.as should be placed so that it can be loaded in TMInterface.
    # Usually Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"
    target_python_link_path = Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"

    # Typically path(os.path.expanduser("~")) / "Documents" / "TrackMania"
    trackmania_base_path = Path(os.path.expanduser("~")) / "Documents" / "TrackMania"

    # Communication port for the first TMInterface instance that will be launched.
    # If using multiple instances, the ports used will be base_tmi_port + 1, +2, +3, etc...
    # This can be left as-is by default
    base_tmi_port = 8478

    # If on Linux, path of a shell script that launches the game, with the TMInterface port as first argument
    linux_launch_game_path = "/mnt/ext4_data/projects/trackmania_rl/scripts/launch_game_pb.sh"
