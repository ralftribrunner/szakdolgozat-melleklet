#!/bin/bash

cd dataset
one=0
zero=0
for filename in *; do
    if [[ $filename =~ _1.jpg ]]; then 
	((one=one+1))	
	cp $filename ../train/pallet/

    elif [[ $filename =~ _0.jpg ]]; then 
	((zero=zero+1))
	cp $filename ../train/empty/
    fi
done

echo $one
echo $zero
