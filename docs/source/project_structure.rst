=================
Project Structure
=================

The full tree structure of the linesight repository is given below:

.. code-block:: bash

    trackmania_rl/
    ├── config_files
    │   ├── config.py
    │   ├── inputs_list.py
    │   ├── state_normalization.py
    │   └── user_config.py
    ├── maps
    │   └── map5_0.5m_cl.npy
    ├── pyproject.toml
    ├── README.md
    ├── requirements_conda.txt
    ├── requirements_pip.txt
    ├── scripts
    │   ├── launch_game_agade.sh
    │   ├── launch_game_pb.sh
    │   ├── tools
    │   │   ├── gbx_to_times_list.py
    │   │   ├── gbx_to_vcp.py
    │   │   ├── tmi1.4.3
    │   │   │   └── move_car_around_test_wall_touch.py
    │   │   ├── tmi2
    │   │   │   ├── add_cp_as_triggers.py
    │   │   │   ├── add_vcp_as_triggers.py
    │   │   │   └── empty_template.py
    │   │   └── video_stuff
    │   │       ├── animate_race_time.py
    │   │       ├── a_v_widget_folder.py
    │   │       └── inputs_to_gbx.py
    │   └── train.py
    ├── setup.py
    └── trackmania_rl
        ├── agents
        │   └── iqn.py
        ├── analysis_metrics.py
        ├── buffer_management.py
        ├── buffer_utilities.py
        ├── contact_materials.py
        ├── experience_replay
        │   └── experience_replay_interface.py
        ├── geometry.py
        ├── map_loader.py
        ├── map_reference_times.py
        ├── multiprocess
        │   ├── collector_process.py
        │   ├── debug_utils.py
        │   └── learner_process.py
        ├── reward_shaping.py
        ├── run_to_video.py
        ├── tmi_interaction
        │   ├── game_instance_manager.py
        │   ├── Python_Link.as
        │   └── tminterface2.py
        └── utilities.py
