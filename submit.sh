#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Gurpreet Singh, gursing\nMark Bzomowski,mbzomows" > submit/team.txt

# make predictions on example data submit it in pred.txt
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# copy all the python files to submit
cp *.py submit/

# make zip file
zip -r submit.zip submit
