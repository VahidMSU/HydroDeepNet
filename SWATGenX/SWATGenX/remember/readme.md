REMEMBER:

we change the import_weather.py to handle an issue with weather stations:

cp -r /usr/local/share/SWATPlusEditor/swatplus-editor/src/api/actions/import_weather.p
y /data/SWATGenXApp/codes/SWATGenX/SWATGenX/remember/

I later find out the range of pcp/tmp and hmd/slr/wnd that come from different dataset must be the same otherwise in some cases it raises an error related to list out of index. 



