# Requirements for Data Preprocessing

### Install DRR Generator
1. Download appropriate version of ITK from https://github.com/InsightSoftwareConsortium/ITK
2. Pass the cmake switch to build ITKTwoProjectionRegistration available in the Modules/Remote. Compile only required Modules. Please choose the ITK version that works with the available C++ compiler. For example, older ITKv4.0 does not work with g++ compiler > 8.0
For example, 
```shell
mkdir external
cd external
wget https://github.com/InsightSoftwareConsortium/ITK/archive/refs/tags/v5.3.0.tar.gz
tar -xzvf https://github.com/InsightSoftwareConsortium/ITK/archive/refs/tags/v5.3.0.tar.gz
cd ITK-5.3.0
mkdir build
cd build
cmake .. -DModule_TwoProjectionRegistration=ON -DBUILD_TESTING=ON -DBUILD_EXAMPLES=ON 
make -j 
```
3. Add the executable to Path

```shell
export PATH="/path/to/bin:$PATH"
```
