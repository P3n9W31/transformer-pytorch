#!/bin/bash

set -o pipefail
set -e

refs=$1
hyp=$2

bleu_script=scripts/multi-bleu.perl
cat_hyp="cat $hyp"
calc_bleu="perl $bleu_script -lc $refs"
bleu=$($cat_hyp | $calc_bleu | cut -f 3 -d ' ' | cut -f 1 -d ',')
echo $bleu
