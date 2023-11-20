#!/bin/bash

cd ~/tmnf_steam/
#sh -c 'echo $$; exec wine ./TMInterfaceTesting.exe'
exec wine "./TMInterfaceTesting.exe" "/configstring=set custom_port $1" #WINEFSYNC=1 STAGING_SHARED_MEMORY=1 STAGING_WRITECOPY=1