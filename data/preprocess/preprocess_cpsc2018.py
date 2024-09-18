import pandas as pd
import numpy as np
import os
from signal_analysis import calculate_waveforms
from tqdm import tqdm
import wfdb
from scipy.io import loadmat
import argparse

def prepare(args):
    data_dir = args.data_dir
    report_list = ["", "normal", "atrial fibrillation", "1 degree atrioventricular block", 
                   "left bundle branch block", "right bundle branch block", "premature atrial contraction", 
                   "premature ventricular contraction", "st-segment depression", "st-segment elevated"]
    sig_name = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    units = ['mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV']
    reference_path = "REFERENCE.csv"
    for base_dir in [os.path.join(data_dir, 'training_set/'), os.path.join(data_dir, 'validation_set/')]:
        df = pd.read_csv(os.path.join(base_dir, reference_path))
        reports = []
        for i in range(len(df)):
            report = report_list[df['First_label'].values[i]]
            if not np.isnan(df['Second_label'].values[i]):
                report += (" "+report_list[int(df['Second_label'].values[i])])
            if not np.isnan(df['Third_label'].values[i]):
                report += (" "+report_list[int(df['Third_label'].values[i])])  
            reports.append(report)
            df['reports'] = reports
        recordings = df.Recording.values

        for re in recordings:
            mat = loadmat(os.path.join(base_dir, re))
            signal = mat['ECG'][0][0][2]
            wfdb.wrsamp(fmt=["32"]*12, adc_gain=[1000]*12, baseline=[0]*12, record_name=re,
                        d_signal=np.array(signal.T*1000, dtype="int"), 
                        sig_name=sig_name, units=units, fs=500, 
                        write_dir=base_dir)
        df.to_csv(os.path.join(base_dir, reference_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="")
    args = parser.parse_args()

    prepare(args)