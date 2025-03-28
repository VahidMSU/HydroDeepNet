
cd /data/SWATGenXApp/codes/Michigan/baseflow/
mkdir -p build
cd build
rm -rf *
source /opt/intel/oneapi/setvars.sh

ifx -c ../source/baseflow.f
ifx -c ../source/caps.f

## now compile
ifx -o baseflow ../source/baseflow.f ../source/caps.f

cp baseflow /data/SWATGenXApp/codes/Michigan/baseflow/