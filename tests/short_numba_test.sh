#!/bin/bash

MYDIR=$(pwd)


cd ../python_script_modules
ADD_PYTHON_DIR=$(pwd)
cd $MYDIR

if ! (echo $PYTHONPATH | tr ':' '\n' | grep -qFx $ADD_PYTHON_DIR)
then
    export PYTHONPATH=$ADD_PYTHON_DIR:$PYTHONPATH
fi

export OMP_NUM_THREADS=2
export OML_NUM_PROCS=2 # The variable says how many processes to use during joblib-parallelized parts.
                        # Note that learning_curve_building.py at init_compound_list uses a dirty hack to assign it equalling OMP_NUM_THREADS.

./recompile_qml.sh
for script in FJK_pair_sep_ibo_kernel_HOMO_numba
do
	python $script.py $script.log
	bench=benchmark_data/$script.dat
	if [ -f $bench ]
	then
		diff_lines=$(diff $script.log $bench | wc -l)
		if [ "$diff_lines" == "0" ]
		then
			echo "Passed: $script"
		else
			echo "Possible Failure: $script"
		fi
	else
		mv $script.log $bench
		echo "Benchmark created: $script"
	fi
done
