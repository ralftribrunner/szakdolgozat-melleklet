#!/bin/bash
#Az adott osztály mappájából áthelyez minden x és 10 közötti elemet.

threshold=$1 # A threshold (x) paraméterként megadható.
cd train/pallet

i=1
for filename in *; do
	if (( $i >= $threshold )); then 
	    mv $filename ../../val/pallet #Áthelyezés a raklap validációs mappájába.
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
	    mv $filename ../../val/empty #Áthelyezés a raklap nélküli képek validációs mappájába.
	fi
	((i=i+1))
	if (( i == 11 )); then 
	   i=1
	fi
done
