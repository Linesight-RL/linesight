============
Installation
============

Prerequisites
-------------

Linesight requires:
    - Python >=3.10 and <3.12
    - PyTorch >=2.1 (check the `official website <https://pytorch.org/get-started/locally/>`_ for specific instructions)
    - a Nvidia GPU with CUDA
    - 20 GB RAM
    - `Trackmania Nations Forever <https://store.steampowered.com/app/11020/TrackMania_Nations_Forever/>`_ with `ModLoader <https://tomashu.dev/software/tmloader/>`_ and `TMInterface 2.1.0 <https://www.donadigo.com/tminterface/>`_.

This project is compatible with Windows and Linux.

Python project setup
--------------------

Clone the repository and install the project.

To avoid outdated package versions, we recommend setting the project up in a clean virtual environment (conda, mamba, pipenv, ...).

conda / mamba
~~~~~~~~~~~~~

If using `conda` or `mamba`, the following commands will set the project up.

.. code-block:: bash

    git clone https://github.com/pb4git/linesight && cd linesight
    conda create -n linesight python=3.11
    conda activate linesight
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # from pytorch website
    conda install --file requirements_conda.txt
    pip install -e .

pip
~~~

In all other cases, use the following commands to set the project in your chosen environment.

.. code-block:: bash

    git clone https://github.com/pb4git/linesight && cd linesight
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # from pytorch website
    pip install -e .


Linux-specific instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the folder ``/scripts/``, create a bash script that takes an integer ``port`` as its single argument. The script should start the game configured to listen on port ``port`` for TMInterface communications. The scripts ``/scripts/launch_game_pb.sh`` and ``/scripts/launch_game_agade.sh`` are working examples on the authors' systems.

.. note::
    The authors have experienced improved FPS when executing TMNF within wine with the following setup:

    .. code-block:: bash

        sudo apt install winetricks
        winetricks dxvk

    and launching the game with ``exec gamemoderun wine (...)`` (see ``/scripts/launch_game_pb.sh`` for example).

Game configuration
------------------

The game must be configured (via ``TmForeverLauncher.exe`` in the game's installation directory) to run in **windowed mode**.
We recommend adjusting game settings to run at the lowest resolution available with low graphics quality.

.. note::
   There is a compromise to be made between *training speed* which increases with FPS and *trained performance* which increases with image quality. Users can experiment with their setup, or use the authors' configuration `available here <_static/authors_settings.png>`_.

User config
-----------

Open the file ``config_files/user_config.py`` and make modifications relevant to your system:

.. code-block:: python

    username = "tmnf_account_username"  # Username of the TMNF account

    # Path where Python_Link.as should be placed so that it can be loaded in TMInterface.
    # Usually Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"
    target_python_link_path = Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"

    # Typically path(os.path.expanduser("~")) / "Documents" / "TrackMania"
    trackmania_base_path = Path(os.path.expanduser("~")) / "Documents" / "TrackMania"

    # Communication port for the first TMInterface instance that will be launched.
    # If using multiple instances, the ports used will be base_tmi_port + 1, +2, +3, etc...
    base_tmi_port = 8478

    # If on Linux, path of a shell script that launches the game, with the TMInterface port as first argument
    linux_launch_game_path = "path_to_be_filled_only_if_on_linux"

    # If on windows, path where TMLoader can be found.
    # Usually Path(os.path.expanduser("~") / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"
    windows_TMLoader_path = Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"

    # If on windows, name of the TMLoader profile that with launch TmForever + TMInterface
    windows_TMLoader_profile_name = "default"
