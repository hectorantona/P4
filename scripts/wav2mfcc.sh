#!/bin/bash

# Base name for temporary files
base=/tmp/$(basename $0).$$

# Ensure cleanup of temporary files on exit
trap cleanup EXIT
cleanup() {
   \rm -f $base.*
}

if [[ $# != 4 ]]; then
   echo "$0 mfcc_order bank_size input.wav output.lp"
   exit 1
fi

mfcc_order=$1
bank_size=$2
inputfile=$3
outputfile=$4

UBUNTU_SPTK=1
if [[ $UBUNTU_SPTK == 1 ]]; then
   # In case you install SPTK using debian package (apt-get)
   X2X="sptk x2x"
   FRAME="sptk frame"
   WINDOW="sptk window"
   LPC="sptk lpc"
   MFCC="sptk mfcc"
else
   # or install SPTK building it from its source
   X2X="x2x"
   FRAME="frame"
   WINDOW="window"
   LPC="lpc"
   MFCC="mfcc"
fi

# Main command for feature extration
sox $inputfile -t raw -e signed -b 16 - | $X2X +sf | $FRAME -l 180 -p 100 | $WINDOW -l 180 -L 180 |
	$MFCC -l 180 -m $mfcc_order -n $bank_size > $base.mfcc

# Our array files need a header with the number of cols and rows:
ncol=$((mfcc_order)) # mfcc
nrow=$($X2X +fa < $base.mfcc | wc -l | perl -ne 'print $_/'$ncol', "\n";')

# Build fmatrix file by placing nrow and ncol in front, and the data after them
echo $nrow $ncol | $X2X +aI > $outputfile
cat $base.mfcc >> $outputfile

exit
