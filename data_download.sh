#!/bin/bash

# Load datasets into specific folder for future use

# location=datasets/cmu_arctic

# if [ ! -e $location ]
# then
#     echo "Create " $location
#     mkdir -p $location
# fi

# root=http://festvox.org/cmu_arctic/packed/

# function download() {
#     identifier=$1
#     file=$2
#     echo "Start downloading $identifier, $file"
#     mkdir -p ${location}/tmp
#     curl -L -o ${location}/tmp/${identifier}.tar.bz2 $file
#     tar xjf ${location}/tmp/${identifier}.tar.bz2 -C ${location}
#     rm -rf ${location}/tmp
# }

# for f in bdl clb rms slt # aew ahw aup awb axb bdl eey fem gka jmk ksp ljm lnh rms rxr slp slt 
# do
#     zipfile=${root}cmu_us_${f}_arctic.tar.bz2
#     download $f $zipfile
# done

## LJSpeech
location=datasets

if [ ! -e $location/tmp ]
then
    echo "Create " ${location}/tmp
    mkdir -p ${location}
fi

# name=LJSpeech-1.1
# wget -o ${location}/tmp/${name}.tar.bz2\
#     -q --show-progress \
#     https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
# tar xjf ${location}/tmp/${name}.tar.bz2 -C ${location}

## VCTK
name=VCTK-Corpus
wget -o ${location}/tmp/${name}.zip\
    -q --show-progress \
    https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip # ?sequence=2&isAllowed=y
unzip ${location}/tmp/${name}.zip -d ${location}
rm -rf ${location}/tmp

