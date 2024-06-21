#!/bin/bash

cd ~/Documents/Programming/TMNF/

# Basic launch command:
#exec wine "./TMInterfaceTesting.exe" "/configstring=set custom_port $1" #WINEFSYNC=1 STAGING_SHARED_MEMORY=1 STAGING_WRITECOPY=1

# Gamemode launch command: may not work on all systems
exec gamemoderun wine ./TMLoader.exe run TmForever "Default" /configstring="set custom_port $1" #WINEFSYNC=1 STAGING_SHARED_MEMORY=1 STAGING_WRITECOPY=1