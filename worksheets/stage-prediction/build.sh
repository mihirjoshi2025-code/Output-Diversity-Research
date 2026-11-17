#!/bin/sh
# Build the worksheet PDF: ./build.sh  (needs pandoc + xelatex)
set -e
cd "$(dirname "$0")"
TEXINPUTS="$PWD/build//:" pandoc stage_prediction_worksheet.md \
  --pdf-engine=xelatex -H build/header.tex --lua-filter build/boxes.lua \
  -o stage_prediction_worksheet.pdf
echo "wrote stage_prediction_worksheet.pdf"
