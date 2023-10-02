from os import path, makedirs
import pandas as pd

from dataset import (
    dome_apic_2022_utils,
)
from dataset.pollen_color_extraction_utils import get_pollen_color_of_images

SELF_PATH = path.dirname(path.realpath(__file__))
CACHE_FOLDER = f"{SELF_PATH}/../tmp/pollen_color_dataset_cache/"
makedirs(CACHE_FOLDER, exist_ok=True)
APIC_DOME_2022_POLLEN_COLOR_CACHE = (
    f"{CACHE_FOLDER}/2022_apic_dome_pollen_color_cache.ftr"
)
APIC_DOME_2022_ONDEVICE_POLLEN_COLOR_CACHE= (
    f"{CACHE_FOLDER}/2022_apic_dome_ondevice_pollen_color_cache.ftr"
)


class PollenColorDataset:
    """
    A class representing a dataset of pollen colors.
    It can be initialized for one subset (e.g. data aquisition run).

    Args:
        subset (str): The data aquisition run to load. Valid values are:
            - "2022_apic_dome": Load data for the 2022_apic_dome subset.

    Attributes:
        subset (dict): A dictionary containing the loaded dataset subset with the following keys:
            - "images": A dictionary containing the different types of illumination.
                - "white": A list of file names corresponding to white illumination.
                - "UV": A list of file names corresponding to UV illumination.
                - "IR": A list of file names corresponding to IR illumination.
            - "pollen_color": Pandas DataFrame containing the pollen colors.
            - "lab_results": Pandas DataFrame containing the pollen sample lab results.

    Examples:
        To load the 2022_apic_dome subset:

        ```python
        dataset = PollenColorDataset("2022_apic_dome")
        ```

        To access the loaded dataset:

        ```python
        images_df = dataset.subset["images"]
        pollen_color_df = dataset.subset["pollen_color"]
        lab_results_df = dataset.subset["lab_results"]
        ```
    """

    def __init__(self, subset):
        self._subset_type = subset
        self.subset = None
        self.load_data()

    def load_data(self):
        if self._subset_type == "2022_apic_dome":
            image = dome_apic_2022_utils.get_images()
            if not path.exists(APIC_DOME_2022_POLLEN_COLOR_CACHE):
                pollen_color = get_pollen_color_of_images(image, True)
                pollen_color_df = pd.concat(list(pollen_color.values())).reset_index()
                pollen_color_df.to_feather(APIC_DOME_2022_POLLEN_COLOR_CACHE)
            else:
                pollen_color_df = pd.read_feather(APIC_DOME_2022_POLLEN_COLOR_CACHE)
            # ondevice data
            if not path.exists(APIC_DOME_2022_ONDEVICE_POLLEN_COLOR_CACHE):
                ondevice_color = dome_apic_2022_utils.get_color_ondevice().reset_index()
                ondevice_color.to_feather(APIC_DOME_2022_ONDEVICE_POLLEN_COLOR_CACHE)
            else:
                ondevice_color = pd.read_feather(APIC_DOME_2022_ONDEVICE_POLLEN_COLOR_CACHE)
            self.subset = {
                "images": image,
                "pollen_color": pollen_color_df,
                "lab_results": dome_apic_2022_utils.get_pollen_sample_lab_results(),
                "on_device_color": ondevice_color
            }
        else:
            # Handle invalid subset
            raise ValueError("Invalid subset specified.")
