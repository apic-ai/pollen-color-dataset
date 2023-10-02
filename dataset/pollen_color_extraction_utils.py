import importlib
import pandas as pd
import sys
import cv2
import os

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

## Utils from the repo pollen color evaluation 2022
try:
    sys.path.append(
        "/Users/langhalsdino/github/apic_ai/software/datascience/publications-and-studies/pollen-color-evaluation-2022/"
    )
    import utils_pollencolor
except Exception as e:
    print(e)
    print(
        "Could not import private utils_pollencolor, is not relevent if cached color exists"
    )


def _crop_img(image, save_crop=False):
    """
    crops the image using the aruco markers
    img: image array
    save_crop: True if the cropped image should be saved
    """
    img_crop = utils_pollencolor.crop_img(image)
    img_crop_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    return img_crop_rgb


def _avg_pollen_color(img, name: str, mask, contours, save=False):
    """
    calculates the mean color value for every pollen and
    img: image as array
    name: name of the image
    mask: binary image mask
    contours: drawn contours
    save: True if images with average pollen color should be saved
    returns
    col_inf: dict with pollen inforamtion
    avg_img: image where every pollen is colored in its mean color
    """
    pollen_prop = utils_pollencolor.pollen_properties(contours, mask, "HQ")
    col_inf = utils_pollencolor.get_color_information(pollen_prop, img, "White")
    avg_img = utils_pollencolor.average_image(col_inf, img.shape, name, save)

    return col_inf, avg_img


def filter_pollen(df):
    """
    filters out pollen which are too small
    df: data frame with pollen information
    """
    # delete rows where pollen area is to small
    df = df[df["areas"] > 500]

    return df


def clean_df(df):
    """
    drops irrelevant columns from the data frame
    """
    df.drop(
        [
            "masks",
            "pixel_HQ",
            "centroids",
            "contours",
            "White_rgb_color",
            "White_lab_color",
        ],
        axis=1,
        inplace=True,
    )
    df.drop(
        [
            "areas",
        ],
        axis=1,
        inplace=True,
    )
    df.rename(
        columns={
            "sample_name": "src_image_name",
            "White_rgb_meanColor": "white_rgb_meanColor",
            "White_lab_meanColor": "white_lab_meanColor",
            "White_rgb_varColor": "white_rgb_varColor",
            "White_lab_varColor": "white_lab_varColor",
        },
        inplace=True,
    )
    df.reset_index(inplace=True, drop=True)

    return df


def get_pollen_color_of_image(image_path, sample_name, save_fig=False):
    print(f"Extracting pollen color for image {image_path} of sample {sample_name}.")
    raw_image = cv2.imread(image_path)
    cropped_image = _crop_img(raw_image)
    mask = utils_pollencolor.background_segment(
        cropped_image, sample_name, save_fig, "segNet"
    )
    contours, _ = utils_pollencolor.pollen_segment(
        mask, cropped_image, sample_name, save_fig
    )
    col_inf, avg_img = _avg_pollen_color(cropped_image, sample_name, mask, contours)

    if save_fig is True:
        if not os.path.isdir(os.path.join(utils_pollencolor.DATA_DEST, "avg_col")):
            os.mkdir(os.path.join(utils_pollencolor.DATA_DEST, "avg_col"))
        utils_pollencolor.save_img(
            f"avg_col/average_{sample_name}.png",
            np.array(avg_img * 255, dtype=np.uint8),
        )

    col_inf["sample_name"] = sample_name
    df_sample = pd.DataFrame(col_inf)
    df_sample = clean_df(filter_pollen(df_sample))

    return df_sample


def get_pollen_color_of_images(image_dataset, save_fig=False):
    pollen_color_dataset = dict()
    print("Extracting pollen color for all images in the dataset.")
    for sample_id in tqdm(image_dataset.keys()):
        samples_pollen_color = []
        for image in tqdm(image_dataset[sample_id]["white"]):
            image_name = image.split("/")[-1]
            pollen_color = get_pollen_color_of_image(image, image_name, save_fig)
            pollen_color["sample_id"] = sample_id
            samples_pollen_color.append(pollen_color)
        pollen_color_dataset[sample_id] = pd.concat(samples_pollen_color)
    return pollen_color_dataset
