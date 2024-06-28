===============
Troubleshooting
===============

**Game stuck on login screen:**

The TMNF account must be an **online account**. If your account is offline:

- Start the game without TMInterface and create an online account
- Set your `username` to that account in `user_config.py`

Linux-specific:
---------------

**Linux installation checklist:**
This list is not exhaustive. It contains the main setup steps the authors use on their machine. It may need to be adapted for your own machine.

1. Update `winehq-staging`
2. Download Steam. Install TMNF from Steam
3. Check that the game can be launched with `wine TmForever.exe` from the installation directory
4. Download the ModLoader zip file, made available on Tomashu's website for linux setups
5. `wine TMLoader.exe` to configure the default profile
6. Check that the game runs with: `wine ~/path/to/TMLoader-win32/TMLoader.exe run TmForever "default" /configstring="set custom_port 8483"`
7. Modify `launch_game_pb.sh` in the repository with the path to ModLoader on your system
8. Install `winetricks`. Apply `winetricks dxvk` for performance.

**Missing OpenAL32.dll**

Install `OpenAL <https://www.openal.org/downloads/>`_ with `wine`.