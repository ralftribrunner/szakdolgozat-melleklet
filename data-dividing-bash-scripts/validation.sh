#!/bin/bash


threshold=$1
cd train/pallet

i=1
for filename in *; do
	if (( $i >= $threshold )); then 
	    mv $filename ../../val/pallet
	fi
	((i=i+1))
	if (( i == 11 )); then 
	   i=1
	fi
done


cd ../empty

i=1
for filename in *; do
	if (( $i >= $threshold )); then 
	    mv $filename ../../val/empty
	fi
	((i=i+1))
	if (( i == 11 )); then 
	   i=1
	fi
done
