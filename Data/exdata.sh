#!/bin/bash
#
# author:	christopher villalpando estrada
# email:	chvillal@ucsc.edu
# last mod:	february 13 2019
#
# Extract Data!
#
# This script will extract and pre-preprocess sample data,
# then store it in the corresponding directories (S1/, S2/, S3/).
#
# Run this script on the small dataset downloaded from:
# https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/
#
# Ideally, place this script on same dir as SMNI_CMI_XXXX.tar.gz
#
# usage: ./exdata.sh <filename> 
#
# example: ./exdata.sh SMNI_CMI_TRAIN.tar.gz

datafile=$1

# check dataset dir was given
if [ "$#" -lt 1 ]; then
	echo "Error! Missing Param"
	echo "Usage: ./exdata.sh SMNI_CMI_TRAIN"
	exit 1
fi

# create output directories
mkdir S1
mkdir S2
mkdir S3

# extrac all subdir files and move them back up
echo; echo "Extracting top ${datafile} ..."
tar -xzf $datafile
datadir=$(echo $datafile| sed 's/\.tar.gz//g')
cd $datadir

#print current dir, for debugging
echo
echo -n "Current Dir: "
pwd
echo

echo "Extracting files ..."; echo

#extract subdirecotry files
gunzip -q */*
mv */* .

#remove empty dirs
find . -type d -empty -delete

echo "Processing files ..."; echo

for filename in *
do
	if [ "$filename" == "README" ]; then
		continue
	fi
	#extract keywords
	class=$( head -n1 ${filename} | grep co | awk '{print $2}'| head -c 4| tail -c 1 )
	type=$(head -n4 $filename | tail -n1 | awk '{print $2 $3}' | sed 's/ //g' )
	if [ "$type" == "S1obj" ]; then
		type="S1"
	elif [ "$type" == "S2match" ]; then
		type="S2"
	elif [ "$type" == "S2nomatch," ]; then
		type="S3"
	fi
	subject=$(head -n1 $filename | grep co | awk '{print $2}'| head -c 11| tail -c 3 )
	trial=$( echo $filename | tail -c 4)
	
	#concatenate a new filename with keywords
	newfilename="${class}_${type}_${subject}_${trial}.csv"

	#remove comments on files
	echo 'trial channel time voltage' > temp
	sed '/#/d' $filename >> temp
	mv temp $newfilename

	#move files to output dirs
	if [ "$type" == "S1" ]; then
		mv $newfilename ../S1/
	elif [ "$type" == "S2" ]; then
		mv $newfilename ../S2/
	else
		mv $newfilename ../S3/
	fi

	#status report!
	echo -n "*"
done

cd ..
rm -rf $datadir

cd S1/
mkdir alcoholic
mkdir control
for filename in *
do
	if [ -d "$filename" ]; then
		continue
	fi

	filter=$( echo $filename | head -c 1 )
	if [ "$filter" == "a" ]; then
		mv $filename alcoholic/
	elif [ "$filter" == "c" ]; then
		mv $filename control/
	fi
done

cd ../S2/
mkdir alcoholic
mkdir control
for filename in *
do
	if [ -d "$filename" ]; then
		continue
	fi

	filter=$( echo $filename | head -c 1 )
	if [ "$filter" == "a" ]; then
		mv $filename alcoholic/
	elif [ "$filter" == "c" ]; then
		mv $filename control/
	fi
done

cd ../S3/
mkdir alcoholic
mkdir control
for filename in *
do
	if [ -d "$filename" ]; then
		continue
	fi

	filter=$( echo $filename | head -c 1 )
	if [ "$filter" == "a" ]; then
		mv $filename alcoholic/
	elif [ "$filter" == "c" ]; then
		mv $filename control/
	fi
done

echo; echo "Done."; echo

