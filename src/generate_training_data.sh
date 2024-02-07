#!/bin/bash

INPUT_DIR=$1
OUTPUT_DIR=$2

if [ "$INPUT_DIR" = "" ] || [ "$OUTPUT_DIR" = "" ]
then
  echo "Usage: $0 <input dir> <output dir>"
  exit
fi

[ -d "$INPUT_DIR" ] || die "Input directory $dir does not exist"
[ -d "$OUTPUT_DIR" ] || die "Output directory $dir does not exist"


# find all direct subdirs of INPUT_DIR
for directory in `find $INPUT_DIR -maxdepth 1 -mindepth 1 -type d `
do
    echo "directory: $directory"
    technique=$(basename $directory)
    ./tag_image.py -o $OUTPUT_DIR -t $technique $directory/*
done