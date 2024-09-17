import pandas as pd
import numpy as np
import os
from signal_analysis import calculate_waveforms
from tqdm import tqdm
import wfdb
import argparse

def prepare(args):
    data_dir = args.data_dir
    df = pd.read_csv(os.path.join(data_dir, "record_list.csv"))
    print("Start calculating waveform data...")
    data_dict = calculate_waveforms(data_dir, df.path.values)
    
    for key in data_dict.keys():
        df[key] = data_dict[key]
        df.to_csv(os.path.join(data_dir, "new_record_list.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="")
    args = parser.parse_args()

    prepare(args)