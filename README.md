# apic.ai - Pollen color dataset

Welcome to the "Pollen color dataset" repository!
This comprehensive dataset offers a collection of color-calibrated images alongside corresponding information about the plants from which the pollen originated. 
The dataset is compiled from pollen traps installed within beehives, capturing pollen over a specific time frame, typically ranging from 24 to 48 hours. 


# Dataset

The pollen color dataset class representing a dataset for each data aquisition run (e.g. subset).
It can be initialized for one subset.

Available subsets:
 - "2022_apic_dome": Load data for the 2022_apic_dome subset.


Each subset hat the following attributes (dictionary):
 - "images": A dictionary containing the different types of illumination.
     - "white": A list of file names corresponding to white illumination.
     - "UV": A list of file names corresponding to UV illumination.
     - "IR": A list of file names corresponding to IR illumination.
 - "pollen_color": Pandas DataFrame containing the pollen colors.
 - "lab_results": Pandas DataFrame containing the pollen sample lab results.

## How to use:

```python3
from dataset.pollen_color_dataset import PollenColorDataset

# create dataset for specific subset
dataset = PollenColorDataset("2022_apic_dome")

# access the loaded dataset
images_df = dataset.subset["images"]
pollen_color_df = dataset.subset["pollen_color"]
ondevice_color_df = datset.subset["on_device_color"]
lab_results_df = dataset.subset["lab_results"]
```

## 00_images

### 2022_apic_dome_images

50 samples from 4 different locations and 10 different hives are available.

![Example image](00_images/2022_apic_dome_images/2022-09-19_10-26-20_hqcam_white_30000_1_rg-3.6-bg-1.55_2.png)

## 00_lab-results

For each sample a dataframe describes the distribution of plants. Each row describes one sample, while the column describe the technical term of the plant. Values the respective proportion in pph.
