from os import path, listdir
from collections import defaultdict
import pandas as pd
import numpy as np
from glob import glob
from ast import literal_eval
from PIL import ImageColor
from skimage import color
import csv


SELF_PATH = path.dirname(path.realpath(__file__))
IMAGE_FOLDER_PATH = f"{SELF_PATH}/../00_images/2022_apic_dome_images/"
ONDEVICE_FOLDER = f"{SELF_PATH}/../00_ondevice/2022_apic_dome_images/"
LAB_RESULTS_PATH = f"{SELF_PATH}/../00_lab-results/2022_apic_dome_images/2022_pollen_analysis_apic-ai_lab_results.csv"
LAB_NAME_MAPPING_PATH = f"{SELF_PATH}/../00_lab-results/2022_apic_dome_images/2022_pollen_analysis_apic-ai_lab_name_mapping.csv"

sep = " ("


SAMPLE_ONDEVICE_HIVE_MAPPING = {
    '2022775': [{'start': pd.Timestamp('2022-05-19 15:00:00'), 'stop': pd.Timestamp('2022-05-20 20:30:00'), 'hive': 'Oc 0'}, {'start': pd.Timestamp('2022-05-19 15:00:00'), 'stop': pd.Timestamp('2022-05-20 20:30:00'), 'hive': 'Oc 1'}],
    '2022773': [{'start': pd.Timestamp('2022-05-19 15:00:00'), 'stop': pd.Timestamp('2022-05-20 20:30:00'), 'hive': 'Oc 2'}],
    '2022774': [{'start': pd.Timestamp('2022-05-19 15:00:00'), 'stop': pd.Timestamp('2022-05-20 20:30:00'), 'hive': 'Oc 3'}],
    '2022779': [{'start': pd.Timestamp('2022-05-30 10:00:00'), 'stop': pd.Timestamp('2022-06-01 16:00:00'), 'hive': 'Oc 0'}],
    '2022776': [{'start': pd.Timestamp('2022-05-30 10:00:00'), 'stop': pd.Timestamp('2022-06-01 16:00:00'), 'hive': 'Oc 1'}],
    '2022777': [{'start': pd.Timestamp('2022-05-30 10:00:00'), 'stop': pd.Timestamp('2022-06-01 16:00:00'), 'hive': 'Oc 2'}],
    '2022778': [{'start': pd.Timestamp('2022-05-30 10:00:00'), 'stop': pd.Timestamp('2022-06-01 16:00:00'), 'hive': 'Oc 3'}],
    '2022782': [{'start': pd.Timestamp('2022-06-14 10:00:00'), 'stop': pd.Timestamp('2022-06-16 20:57:00'), 'hive': 'Oc 0'}],
    '2022780': [{'start': pd.Timestamp('2022-06-14 10:00:00'), 'stop': pd.Timestamp('2022-06-16 21:00:00'), 'hive': 'Oc 1'}],
    '2022781': [{'start': pd.Timestamp('2022-06-14 10:00:00'), 'stop': pd.Timestamp('2022-06-16 21:15:00'), 'hive': 'Oc 2'}],
    '2022786': [{'start': pd.Timestamp('2022-07-04 17:00:00'), 'stop': pd.Timestamp('2022-07-06 15:00:00'), 'hive': 'Oc 0'}],
    '2022783': [{'start': pd.Timestamp('2022-07-04 17:00:00'), 'stop': pd.Timestamp('2022-07-06 14:57:00'), 'hive': 'Oc 1'}],
    '2022784': [{'start': pd.Timestamp('2022-07-04 17:00:00'), 'stop': pd.Timestamp('2022-07-06 14:53:00'), 'hive': 'Oc 2'}],
    '2022785': [{'start': pd.Timestamp('2022-07-04 17:00:00'), 'stop': pd.Timestamp('2022-07-06 14:50:00'), 'hive': 'Oc 3'}],
    '2022787': [{'start': pd.Timestamp('2022-07-12 10:00:00'), 'stop': pd.Timestamp('2022-07-14 16:00:00'), 'hive': 'Oc 0'}],
    '2022788': [{'start': pd.Timestamp('2022-07-26 10:00:00'), 'stop': pd.Timestamp('2022-07-28 16:00:00'), 'hive': 'Oc 2'}],
    '2022789': [{'start': pd.Timestamp('2022-07-26 10:00:00'), 'stop': pd.Timestamp('2022-07-28 16:00:00'), 'hive': 'Oc 3'}],
    '2022791': [{'start': pd.Timestamp('2022-08-23 10:00:00'), 'stop': pd.Timestamp('2022-08-25 16:00:00'), 'hive': 'Oc 2'}],
    '2022792': [{'start': pd.Timestamp('2022-08-23 10:00:00'), 'stop': pd.Timestamp('2022-08-25 16:00:00'), 'hive': 'Oc 3'}],
    '2022795': [{'start': pd.Timestamp('2022-09-06 10:00:00'), 'stop': pd.Timestamp('2022-09-08 16:00:00'), 'hive': 'Oc 0'}],
    '2022793': [{'start': pd.Timestamp('2022-09-06 10:00:00'), 'stop': pd.Timestamp('2022-09-08 16:00:00'), 'hive': 'Oc 2'}],
    '2022794': [{'start': pd.Timestamp('2022-09-06 10:00:00'), 'stop': pd.Timestamp('2022-09-08 16:00:00'), 'hive': 'Oc 3'}],
    '2022796': [{'start': pd.Timestamp('2022-09-27 10:00:00'), 'stop': pd.Timestamp('2022-09-29 16:25:00'), 'hive': 'Oc 1'}],
    '2022797': [{'start': pd.Timestamp('2022-09-27 10:00:00'), 'stop': pd.Timestamp('2022-09-29 16:30:00'), 'hive': 'Oc 2'}],
    '2022798': [{'start': pd.Timestamp('2022-09-27 10:00:00'), 'stop': pd.Timestamp('2022-09-29 16:20:00'), 'hive': 'Oc 3'}],
    '2022820': [{'start': pd.Timestamp('2022-10-12 10:00:00'), 'stop': pd.Timestamp('2022-10-14 16:00:00'), 'hive': 'Oc 0'}],
    '2022817': [{'start': pd.Timestamp('2022-10-12 10:00:00'), 'stop': pd.Timestamp('2022-10-14 16:00:00'), 'hive': 'Oc 1'}],
    '2022818': [{'start': pd.Timestamp('2022-10-12 10:00:00'), 'stop': pd.Timestamp('2022-10-14 16:00:00'), 'hive': 'Oc 2'}],
    '2022819': [{'start': pd.Timestamp('2022-10-12 10:00:00'), 'stop': pd.Timestamp('2022-10-14 16:00:00'), 'hive': 'Oc 3'}],
    '2022821': [{'start': pd.Timestamp('2022-10-27 10:00:00'), 'stop': pd.Timestamp('2022-10-29 13:40:00'), 'hive': 'Oc 2'}],
    '2022822': [{'start': pd.Timestamp('2022-10-27 10:00:00'), 'stop': pd.Timestamp('2022-10-29 13:40:00'), 'hive': 'Oc 3'}],
    '2022799': [{'start': pd.Timestamp('2022-07-19 22:01:00'), 'stop': pd.Timestamp('2022-07-21 18:00:00'), 'hive': 'H 1'}, {'start': pd.Timestamp('2022-07-19 22:01:00'), 'stop': pd.Timestamp('2022-07-21 18:00:00'), 'hive': 'H 2'}],
    '2022800': [{'start': pd.Timestamp('2022-07-26 17:36:00'), 'stop': pd.Timestamp('2022-07-28 11:13:00'), 'hive': 'H 2'}, {'start': pd.Timestamp('2022-07-26 17:36:00'), 'stop': pd.Timestamp('2022-07-28 11:13:00'), 'hive': 'H 1'}],
    '2022801': [{'start': pd.Timestamp('2022-08-09 16:00:00'), 'stop': pd.Timestamp('2022-08-11 17:30:00'), 'hive': 'H 1'}, {'start': pd.Timestamp('2022-08-09 16:00:00'), 'stop': pd.Timestamp('2022-08-11 17:30:00'), 'hive': 'H 2'}],
    '2022802': [{'start': pd.Timestamp('2022-08-23 10:00:00'), 'stop': pd.Timestamp('2022-08-25 16:00:00'), 'hive': 'H 1'}, {'start': pd.Timestamp('2022-08-23 10:00:00'), 'stop': pd.Timestamp('2022-08-25 16:00:00'), 'hive': 'H 2'}],
    '2022803': [{'start': pd.Timestamp('2022-09-06 10:00:00'), 'stop': pd.Timestamp('2022-09-08 16:00:00'), 'hive': 'H 1'}, {'start': pd.Timestamp('2022-09-06 10:00:00'), 'stop': pd.Timestamp('2022-09-08 16:00:00'), 'hive': 'H 2'}],
    '2022804': [{'start': pd.Timestamp('2022-09-21 10:30:00'), 'stop': pd.Timestamp('2022-09-23 16:00:00'), 'hive': 'H 1'}, {'start': pd.Timestamp('2022-09-21 10:30:00'), 'stop': pd.Timestamp('2022-09-23 16:00:00'), 'hive': 'H 2'}],
    '2022816': [{'start': pd.Timestamp('2022-09-04 12:00:00'), 'stop': pd.Timestamp('2022-10-06 09:46:00'), 'hive': 'H 1'}, {'start': pd.Timestamp('2022-09-04 12:00:00'), 'stop': pd.Timestamp('2022-10-06 09:50:00'), 'hive': 'H 2'}],
    '2022807': [{'start': pd.Timestamp('2022-08-09 08:25:00'), 'stop': pd.Timestamp('2022-08-11 08:00:00'), 'hive': 'V 2'}],
    '2022806': [{'start': pd.Timestamp('2022-08-22 09:00:00'), 'stop': pd.Timestamp('2022-08-24 08:30:00'), 'hive': 'V 2'}],
    '2022805': [{'start': pd.Timestamp('2022-09-07 10:00:00'), 'stop': pd.Timestamp('2022-09-09 08:40:00'), 'hive': 'V 2'}],
    '2022810': [{'start': pd.Timestamp('2022-08-10 07:55:00'), 'stop': pd.Timestamp('2022-08-12 07:55:00'), 'hive': 'C 1'}, {'start': pd.Timestamp('2022-08-10 08:00:00'), 'stop': pd.Timestamp('2022-08-12 07:55:00'), 'hive': 'C 2'}],
    '2022809': [{'start': pd.Timestamp('2022-08-23 08:55:00'), 'stop': pd.Timestamp('2022-08-25 08:40:00'), 'hive': 'C 1'}, {'start': pd.Timestamp('2022-08-23 08:55:00'), 'stop': pd.Timestamp('2022-08-25 08:40:00'), 'hive': 'C 2'}],
    '2022808': [{'start': pd.Timestamp('2022-09-11 10:00:00'), 'stop': pd.Timestamp('2022-09-13 15:25:00'), 'hive': 'C 1'}, {'start': pd.Timestamp('2022-09-11 10:00:00'), 'stop': pd.Timestamp('2022-09-13 15:25:00'), 'hive': 'C 2'}]
}


HIVE_LB_MAPPING = {
    'Oc 0': ["LB-JET22-002", "LB-JET22-069", "LB-JET22-014"],
    'Oc 1': ["LB-JET22-001", "LB-JET22-065"],
    'Oc 2': ["LB-JET22-015", "LB-JET22-059"],
    'Oc 3': ["LB-JET22-010"],
    'H 1': ["LB-JET22-035", "LB-JET22-068", "LB-JET22-070"],
    'H 2': ["LB-JET22-036"],
    'V 1': ["LB-JET22-071"],
    'V 2': ["LB-JET22-072"],
    'C 1': ["LB-JET22-073"],
    'C 2': ["LB-JET22-074"],
}



def get_images():
    samples = dict()
    lab_results = pd.read_csv(LAB_NAME_MAPPING_PATH)
    available_images = list(
        filter(lambda image: image.endswith(".png"), listdir(IMAGE_FOLDER_PATH))
    )

    for pollen_sample in lab_results["Prüfnummer "].unique().tolist():
        sample = lab_results[lab_results["Prüfnummer "] == pollen_sample]
        file_names = (
            pd.Series(
                sample["file_name"].values.tolist()
                + sample["file_name_2"].values.tolist()
            )
            .dropna()
            .tolist()
        )
        file_names = [
            f"{IMAGE_FOLDER_PATH}/{file_name}.png"
            for file_name in file_names
            if f"{file_name}.png" in available_images
        ]

        if len(file_names) > 0:
            # split images into UV, IR, and White
            images = {
                "white": list(filter(lambda image: "hqcam_white" in image, file_names)),
                "UV": list(filter(lambda image: "hqcam_uv" in image, file_names)),
                "IR": list(filter(lambda image: "hqcam_ir" in image, file_names)),
            }

            samples[pollen_sample] = images
    return samples


def has_header(csv_file_path, delimiter=','):
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        first_row = next(reader)
        return first_row[0] == 'insect_id'


def contains_pollen_color(csv_file_path, delimiter=','):
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        try:
            first_row = next(reader)
            if len(first_row) > 10:
                print(f"WARNING: {csv_file_path} has more than 10 columns.")
            return (
                'pollen_color' in first_row
            ) or (
                len(first_row) == 10
                and not has_header(csv_file_path, delimiter=delimiter)
            )
        except StopIteration:
            return False

def is_18_fps(csv_file_path, delimiter=','):
    if path.exists(f"{csv_file_path}.uptime.csv"):
        with open(f"{csv_file_path}.uptime.csv", 'r') as file:
            reader = csv.reader(file, delimiter=delimiter)
            try:
                first_row = next(reader)
                second_row = next(reader)
                if second_row[-2] == "1080":
                    return True
                elif second_row[-2] == "1200":
                    return False
                else:
                    print(f"WARNING: {csv_file_path}.uptime.csv has unknown resolution {second_row[-2]}")
                    return False
            except Exception as e:
                print(f"WARNING: {csv_file_path}.uptime.csv is empty. {e}")
                return False
    else:
        return False

def get_color_ondevice():
    MIN_Y_DIST_INGRESS = 50
    MIN_POLLEN_FRACTION = 0.3
    MIN_POLLEN_FOUND = 3
    samples = []
    for sample in SAMPLE_ONDEVICE_HIVE_MAPPING:
        for hive_sample in SAMPLE_ONDEVICE_HIVE_MAPPING[sample]:
            start = hive_sample['start'].timestamp()
            stop = hive_sample['stop'].timestamp()
            if hive_sample['hive'] in HIVE_LB_MAPPING:
                hive = hive_sample['hive']
                for device_name in HIVE_LB_MAPPING[hive]:
                    device_data_path = path.join(ONDEVICE_FOLDER, f"ondevice-{hive.split(' ')[0].lower()}", device_name)
                    if path.exists(device_data_path):
                        epochs = []
                        for csv_file in glob("**/*bee_[0-9]*.csv", root_dir=device_data_path, recursive=True):
                            if not 'uptime' in csv_file:
                                epochs.append(int(csv_file.split('_')[-1].split('.')[0]))
                        epochs = [epoch for epoch in epochs if epoch <= stop]
                        epochs = sorted(epochs)
                        if len(epochs) > 0:
                            # we want the list to include all epochs > start and the one before start
                            try:
                                index = next(i for i, e in enumerate(epochs) if e > start)
                                if index > 0:
                                    epochs = epochs[index-1:]
                                else:
                                    epochs = epochs[index:]
                                dfs = []
                                for epoch in epochs:
                                    for p in glob(f"**/*bee_{epoch}.csv", root_dir=device_data_path, recursive=True):
                                        if contains_pollen_color(path.join(device_data_path, p)):
                                            if has_header(path.join(device_data_path, p)):
                                                df = pd.read_csv(
                                                    path.join(device_data_path, p),
                                                    sep=','
                                                )
                                            else:
                                                df = pd.read_csv(
                                                    path.join(device_data_path, p),
                                                    sep=',',
                                                    header=None,
                                                    names=['insect_id', 'first_frame_id', 'start_point_x', 'start_point_y', 'end_point_x', 'end_point_y',
                                                        'path_duration', 'pollen_found', 'no_pollen_found', 'pollen_color']
                                                )
                                            # only use dataframes with pollen_color available
                                            if 'pollen_color' in df.columns:
                                                if is_18_fps(path.join(device_data_path, p)):
                                                    df['timestamp'] = (df['first_frame_id'] / 18 + epoch).apply(lambda x: pd.Timestamp(x, unit='s'))
                                                else:
                                                    df['timestamp'] = (df['first_frame_id'] / 20 + epoch).apply(lambda x: pd.Timestamp(x, unit='s'))
                                                # Remove all bees without pollen
                                                df = df[df['pollen_color'] != '[]']
                                                df = df[df['pollen_found'] >= MIN_POLLEN_FOUND]
                                                df = df[(df['end_point_y'] - df['start_point_y']) > MIN_Y_DIST_INGRESS]
                                                df = df[(df['pollen_found'] / (df['pollen_found'] + df['no_pollen_found'])) >= MIN_POLLEN_FRACTION]
                                                # remove all incorect color values
                                                df['pollen_color'] = df.pollen_color.apply(lambda x: [c for c in [item for sublist in literal_eval(x) for item in sublist] if len(c) == 6])
                                                df['pollen_color'] = df.pollen_color.apply(lambda x: [color.rgb2lab(np.array(ImageColor.getcolor('#'+c, 'RGB'))/255) for c in x])
                                                df['mean_lab_pollen_color'] = df.pollen_color.apply(lambda x: np.mean(x, axis=0))
                                                df['median_lab_pollen_color'] = df.pollen_color.apply(lambda x: np.median(x, axis=0))
                                                df['var_lab_pollen_color'] = df.pollen_color.apply(lambda x: np.var(x, axis=-1))
                                                # only keep relevant columns
                                                df = df[['timestamp', 'mean_lab_pollen_color', 'median_lab_pollen_color', 'var_lab_pollen_color']]
                                                df['sample_id'] = sample
                                                df['hive'] = hive
                                                if len(df) > 0:
                                                    dfs.append(df)
                                        else:
                                            print(f"WARNING: {p} does not contain pollen_color")
                                if len(dfs) > 0:
                                    df = pd.concat(dfs)
                                    df = df[(df['timestamp'] <= pd.Timestamp(stop, unit='s')) & (df['timestamp'] >= pd.Timestamp(start, unit='s'))]
                                    samples.append(df)
                            # If no valid epoch is available sp
                            except StopIteration:
                                pass
            else:
                print(f"missing data for {hive_sample}")
    return pd.concat(samples).reset_index()

def get_pollen_sample_lab_results():
    df = pd.read_csv(LAB_RESULTS_PATH)
    new_columns = []
    for column_name in df.columns:
        if sep in column_name:
            new_columns.append(column_name.split(sep)[0])
        else:
            new_columns.append(column_name)

    df = df.rename(columns=dict(zip(df.columns, new_columns)))
    df.columns = df.columns.str.replace(" Gänsefuß", "")
    df.columns = [col.rstrip() if col.endswith(" ") else col for col in df.columns]

    df = df.rename(columns={"PR_ID": "sample_id", "unbekannt": "unknown"})
    df.drop(columns=["zz", "Gesamtsumme"], inplace=True)

    for column in df.columns:
        if column != "sample_id" and df[column].dtype == object:
            df[column] = df[column].str.replace(",", ".").astype(float)

    return df
