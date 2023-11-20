#!/bin/bash

cd ~/Documents/Programming/TMNF/
#sh -c 'echo $$; exec wine ./TMInterfaceTesting.exe'
exec wine "./TMInterfaceTesting.exe" "/configstring=set custom_port $1" #WINEFSYNC=1 STAGING_SHARED_MEMORY=1 STAGING_WRITECOPY=1