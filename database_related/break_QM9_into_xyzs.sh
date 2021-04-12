#!/bin/bash
# QM9's latest available version has all molecule xyz's merged into a single xyz file.
# This is the script I used to break it down into individual molecular xyzs which is
# more convenient for scripting.


input=$1

if [ "$input" == "" ]
then
    echo "Specify input file."
    exit
fi

xyz_folder="xyz_folder"
mkdir -p $xyz_folder
cd $xyz_folder

awk '
{
if (filename == ""){
    counter++;
    filename=sprintf("dsgdb9nsd_%06d.xyz", counter);
};
if ($0 == ""){
    filename=""
}
else {
    print $0 >> filename    
}
}
' ../$input
