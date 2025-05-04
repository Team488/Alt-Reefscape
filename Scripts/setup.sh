#! /bin/bash
set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

if ! [[ -x $(command -v pyenv) ]]; then
    echo "Pyenv must be installed."
    exit 1
fi

pyenv virtualenv --version

if [[ $(pyenv versions | grep 3.10.16 | wc -l) == "0" ]]; then
    pyenv install 3.10.16
fi

if [[ $(pyenv versions | grep xbot-3.9 | wc -l) == "0" ]]; then
    pyenv virtualenv 3.10.16 xbot-3.10
fi

eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate xbot-3.10

if ! [[ $(grep -c "export PYTHONPATH=$PYTHONPATH:$(realpath $SCRIPT_DIR)/src" $VIRTUAL_ENV/bin/activate) -ge 1 ]]; then
    cat <<EOF >> $VIRTUAL_ENV/bin/activate
export PYTHONPATH=$PYTHONPATH:$(realpath $SCRIPT_DIR)/src
EOF

pyenv deactivate xbot-3.10
pyenv activate xbot-3.10
fi

export PYTHONPATH=$PYTHONPATH:$(realpath $SCRIPT_DIR)/src

pip install -r $SCRIPT_DIR/requirements-py-3-10.txt
pip install -r $SCRIPT_DIR/dev-requirements.txt
pip install -r $SCRIPT_DIR/non-base-requirements.txt
pip install --upgrade tensorflow
pip install --upgrade XTablesClient

pre-commit install
pyenv rehash
