import pandas as pd
import numpy as np
import os
from signal_analysis import calculate_waveforms
from tqdm import tqdm
import wfdb
import h5py
import tarfile
import argparse


def prepare(args):
    data_dir = args.data_dir

    if not os.path.exists(os.path.join(data_dir, 'records/')):
        filename = os.path.join(data_dir, 'records.tar.gz')
        tf = tarfile.open(filename)
        tf.extractall(os.path.join(data_dir, 'records/'))
    out_dir = os.path.join(data_dir, "records/records_wfdb")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    base_dir = os.path.join(data_dir, "records/records/")
    files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    out_dir = os.path.join(data_dir, "records/records_wfdb")
    sig_name = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    units = ['mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV']

    print("Start converting to WFDB format...")
    for file in tqdm(files):
        with h5py.File(file, "r") as f:
            signal = f['ecg'][:,:]
            wfdb.wrsamp(fmt=["32"]*12, adc_gain=[1000]*12, baseline=[0]*12, record_name=file.split("/")[-1].split(".")[0],
                    d_signal=np.array(signal.T*1000, dtype="int"), 
                    sig_name=sig_name, units=units, fs=500, 
                    write_dir=out_dir)
            
    df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    print("Start calculating waveform data...")
    data_dict = calculate_waveforms(out_dir, df.ECG_ID.values)
    for key in data_dict.keys():
        df[key] = data_dict[key]
        
    df.to_csv(os.path.join(data_dir, "metadata.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="")
    args = parser.parse_args()

    prepare(args)