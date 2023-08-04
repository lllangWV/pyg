import os
import json
from glob import glob
import logging.config
from pathlib import Path


# Other Constants
FILE = Path(__file__).resolve()
ROOT = str(FILE.parents[1])  # pyprocar



data_dir=os.path.join(ROOT,'data','raw','test')

data={}
files=glob(data_dir + '/*.json')
for file in files:
    # print(file)
    filename=file.split(os.sep)[-1].split('.')[0]
    with open(file,'r') as f:
        tmp = json.load(f)

    data.update({filename:tmp})

TEST_DATA=data

