#!/bin/bash

# A meglévő adathalmazt csoportosítja külön mappába az osztálya (raklap/üres) szerint.
cd dataset
one=0
zero=0
for filename in *; do
    if [[ $filename =~ _1.jpg ]]; then #Ha _1.jpg-t tartalmaz a fájlneve.
	((one=one+1))	
	cp $filename ../train/pallet/ #Másolás a raklapos mappába

    elif [[ $filename =~ _0.jpg ]]; then #Ha _0.jpg-t tartalmaz a fájlneve.
	((zero=zero+1))
	cp $filename ../train/empty/ #Másolás az üres mappába
    fi
done

echo $one
echo $zero
