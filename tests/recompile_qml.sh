#!/bin/bash

MYDIR=$(pwd)
package_name=qml
rm -Rf "$(pip show --verbose qml | awk '{if ($1=="Location:") print $2}')/$package_name"
cd ..
rm -f __pycache__/*
rm -Rf build/*
rm -Rf qml/__pycache__/*
python setup.py build
python setup.py install
cd $MYDIR
