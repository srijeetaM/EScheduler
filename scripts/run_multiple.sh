#!/usr/bin/env bash

for i in *.stats
do
	echo "first: $i"
	python revise_dispatch_history.py "$i" ../dag_history/"${i/dispatch/dag}" trace/"$i"
	echo "last: $i"
done
