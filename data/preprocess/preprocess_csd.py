import pandas as pd
import numpy as np
import os
from signal_analysis import calculate_waveforms
from tqdm import tqdm
import argparse
import wfdb


def prepare(args):
    data_dir = args.data_dir
    df = pd.read_csv(os.path.join(data_dir, "ConditionNames_SNOMED-CT.csv"))
    snomed_dict = {}
    for i in range(len(df)):
        snomed_dict[str(df["Snomed_CT"].values[i])] = df['Full Name'].values[i]
        
    snomed_dict["74390002"] = "Wolff-Parkinson-White pattern"
    snomed_dict["55827005"] = "left ventricle hypertrophy"
    snomed_dict["67751000119106"] = "Right atrial enlargement"
    snomed_dict["251166008"] = "Atrioventricular nodal re-entry tachycardia"
    snomed_dict["17366009"] = "Atrial arrhythmia"
    snomed_dict["713427006"] = "complete right bundle branch block"
    snomed_dict["713426002"] = "incomplete right bundle branch block"
    snomed_dict["55930002"] = "ST segment changes"
    snomed_dict["29320008"] = "Ectopic rhythm"
    snomed_dict["10370003"] = "Rhythm from artificial pacing"
    snomed_dict["365413008"] = "R wave"
    snomed_dict["251223006"] = "Tall P wave"
    snomed_dict["6374002"] = "Bundle branch block"
    snomed_dict["81898007"] = "Ventricular escape rhythm"
    snomed_dict["427172004"] = "premature ventricular contractions"
    snomed_dict["445118002"] = "Left anterior fascicular block"
    snomed_dict["425856008"] = "paroxysmal ventricular tachycardia"
    snomed_dict["733534002"] = "complete left bundle branch block"
    snomed_dict["106068003"] = "Atrial rhythm"
    snomed_dict["50799005"] = "Atrioventricular dissociation"
    snomed_dict["61721007"] = "Counterclockwise vectorcardiographic loop"
    snomed_dict["57054005"] = "Acute myocardial infarction"
    snomed_dict["426627000"] = "bradycardia"
    snomed_dict["54329005"] = "Acute myocardial infarction of anterior wall"
    snomed_dict["251205003"] = "Prolonged P wave"
    snomed_dict["61277005"] = "Accelerated idioventricular rhythm"
    snomed_dict["164896001"] = "ventricular fibrillation"
    snomed_dict["111288001"] = "Ventricular flutter"
    snomed_dict["233892002"] = "Ectopic atrial tachycardia"
    snomed_dict["251120003"] = "Incomplete left bundle branch block"
    snomed_dict["251170000"] = "Blocked premature atrial contraction"
    snomed_dict["251187003"] = "Atrial escape complex"
    snomed_dict["418818005"] = "Brugada syndrome"
    snomed_dict["426183003"] = "Mobitz type II atrioventricular block"
    snomed_dict["426648003"] = "junctional tachycardia"
    snomed_dict["426664006"] = "accelerated junctional rhythm"
    snomed_dict["445211001"] = "Left posterior fascicular block"
    snomed_dict["446813000"] = "Left atrial hypertrophy"
    snomed_dict["49578007"] = "Shortened PR interval"
    snomed_dict["5609005"] = "Sinus arrest"
    snomed_dict["63593006"] = "Supraventricular premature beats"
    snomed_dict["65778007"] = "Sinoatrial block"
    snomed_dict["67741000119109"] = "Left atrial enlargement"
    snomed_dict["77867006"] = "Shortened QT interval"

    paths = []
    ages = []
    sexes = []
    reports = []
    codes = []
    nc = []
    file_list = []
    for f in os.walk(os.path.join(data_dir, "WFDBRecords")):
        if(len(f[2]) < 0):
            continue
        for file in f[2]:
            if "mat" in file:
                file_list.append(os.path.join(f[0], file))

    print("Start extracting labels from .hea files...")
    for file_name in tqdm(file_list):
        full_path = os.path.join(f[0], file_name).replace(".mat", "")
        record = wfdb.rdsamp(full_path)[1]
        sn_code = record['comments'][2][4:].replace(',', ';')
        report = " ".join(snomed_dict[code] for code in sn_code.split(';'))
        codes.append(sn_code)
        reports.append(report)
        ages.append(record['comments'][0][5:])
        sexes.append(record['comments'][1][5:])
        paths.append(full_path.split(os.path.join(data_dir))[1])
    diagnostic_df = pd.DataFrame()
    diagnostic_df["filename"] = paths
    diagnostic_df["age"] = ages
    diagnostic_df["sex"] = sexes
    diagnostic_df["report"] = reports
    diagnostic_df["Snomed_CT"] = codes
    diagnostic_df["strat_fold"] = np.array(range(0, len(paths))) % 10
    
    print("Start calculating waveform data...")
    data_dict = calculate_waveforms(data_dir, paths)

    for key in data_dict.keys():
        df[key] = data_dict[key]
    df.to_csv(os.path.join(data_dir, "diagnostics.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="")
    args = parser.parse_args()

    prepare(args)