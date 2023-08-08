

# install venv (in case it is not installed)
sudo apt-get update
sudo apt-get install python3-venv

#create virtual environment
python -m venv LA2_env

#activate virtual environment
source ./LAassignment2_env/bin/activate

#github
bash git_auth.sh

#intall the model
python3 -m spacy download en_core_web_md

#install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# deactivate environment
#deactivate

