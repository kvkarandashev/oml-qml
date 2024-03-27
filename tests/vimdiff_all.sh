#!/bin/bash

for l in *.log # get all output log files
do
	b=benchmark_data/$(echo $l | cut -f1 -d'.').dat # corresponding benchmark file
	vimdiff $l $b # compare
done
