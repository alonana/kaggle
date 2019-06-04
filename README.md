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
pip install xgboost
pip install statsmodels
pip install Pillow
pip install scikit-image

# What have I learned?

- remove outliers
- remove low importance features
- in case we have 2 related features: 1st_floor_size, 2nd_floor_size, 
create additional new feature: total_size
- in case we have a numeric feature, where value zero exists basement_size, 
create additional indication feature: has_basement
- in case of missing values, do not just replace with mean.
First check what is the percent of the missing values.
In some cases, you will assign a special value for the missing values, like 'None'.
In other cases, deduct one of the category values (for example the most common value).
For numeric, you might use the mean value or zero value.

# TODO
- stacking machine learning models