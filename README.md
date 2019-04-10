# Prerequisites

sudo apt install python3-pip
sudo pip3 install -U virtualenv


virtualenv --system-site-packages -p python3 ./venv
OR
In pycharm - create virtual env

source venv/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow
pip install matplotlib
pip install pandas
sudo apt-get install python3-tk
pip install scikit-learn
pip install seaborn


# What have I learned?

- remove outliers
- remove low importance features
- in case we have 2 related features: 1st_floor_size, 2nd_floor_size, 
create additional new feature: total_size
- in case we have a numeric feature, where value zero exists basement_size, 
create additional indication feature: has_basement

# TODO
- stacking machine learning models