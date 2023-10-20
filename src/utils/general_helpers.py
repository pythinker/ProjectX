from datetime import datetime
import random

import pytz
from pytz import timezone


def get_pst_time():

    date_format="%d-%m-%Y--%H-%M-%S"
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(timezone('US/Pacific'))
    pstDateTime=date.strftime(date_format)

    return pstDateTime


def get_tokens(tokens_path):

    with open(tokens_path, "rb") as f:
        data = f.readlines()
    
    tokens = []
    file_names = []
    for line in data:
        file_name = str(line)[2:-3].split('|')[0]
        text = str(line)[2:-3].split('|')[1]
        tokens_ = text.split(', ')
        tokens_ = [int(token) for token in tokens_]
        file_names.append(file_name)
        tokens.append(tokens_)

    return file_names, tokens
