#!/bin/bash

MYDIR=$(pwd)
package_name=qml
if [ -z "$(pip show $package_name 2>&1 >/dev/null)" ]
then
    rm -Rf "$(pip show --verbose $package_name | awk '{if ($1=="Location:") print $2}')/$package_name"
fi
cd ..
rm -f __pycache__/*
rm -Rf build/*
rm -Rf qml/__pycache__/*
python setup.py build
python setup.py install
cd $MYDIR
