#!/usr/bin/env bash

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv deactivate xbot-3.9 || true
pyenv activate xbot-3.9

export PYTHONPATH=$PYTHONPATH:$(realpath $SCRIPT_DIR)/src

python3 $@
