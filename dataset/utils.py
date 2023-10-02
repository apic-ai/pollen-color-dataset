from collections import Counter
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import color
import pandas as pd
from scipy.spatial.distance import cdist


# ID, R, G, B
COLOR_MATCHING = [
    ("Color 1", 0, 0, 0),
    ("Color 2", 38, 51, 74),
    ("Color 3", 74, 25, 44),
    ("Color 4", 94, 33, 41),
    ("Color 5", 117, 21, 30),
    ("Color 6", 142, 64, 42),
    ("Color 7", 204, 6, 5),
    ("Color 8", 211, 110, 112),
    ("Color 9", 244, 219, 170),
    ("Color 10", 250, 250, 210),
    ("Color 11", 255, 255, 209),
    ("Color 12", 255, 216, 127),
    ("Color 13", 255, 255, 151),
    ("Color 14", 240, 240, 140),
    ("Color 15", 248, 243, 53),
    ("Color 16", 255, 215, 0),
    ("Color 17", 255, 165, 0),
    ("Color 18", 237, 118, 14),
    ("Color 19", 252, 62, 2),
    ("Color 20", 220, 156, 0),
    ("Color 21", 175, 117, 5),
    ("Color 22", 159, 80, 9),
    ("Color 23", 117, 75, 25),
    ("Color 24", 146, 50, 0),
    ("Color 25", 157, 145, 1),
    ("Color 26", 198, 196, 100),
    ("Color 27", 202, 196, 176),
    ("Color 28", 126, 123, 82),
    ("Color 29", 53, 124, 45),
    ("Color 30", 80, 78, 0)
]


def cluster_pollen(df, data_column: str, min_cluster_size=10, noise_aware=True):
    """
    clustering of pollen color distribution
    df: data frame with pollen information
    data_column: data on which to perform clustering e.g. 'White_lab_meanColor'
    min_cluster_size: min number of datapoints for a cluster
    noise_aware: False if each outlier (label -1) should be considered as individual cluster

    returns a list of labels
    """
    df = df.copy()
    data = df[data_column]

    data = np.array(data.values.tolist())
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(data)
    labels = list(clusterer.labels_)

    if not noise_aware:
        number_cluster = clusterer.labels_.max()
        for i, label in enumerate(labels):
            if label == -1:
                number_cluster += 1
                labels[i] = number_cluster

    return labels

def static_cluster_pollen(df, lab_column: str, color_matching=None):
    if color_matching is None:
        color_matching_df = pd.DataFrame(COLOR_MATCHING, columns=["ID", "R", "G", "B"])
        color_matching_df["lab"] = color_matching_df.apply(lambda x: color.rgb2lab([x.R / 255, x.G / 255, x.B / 255]), axis=1)
    else:
        color_matching_df = color_matching
        color_matching_df["lab"] = color_matching_df.apply(lambda x: [x.l, x.a, x.b], axis=1)

    distances = cdist(df[lab_column].tolist(), color_matching_df["lab"].tolist())
    closest_indices = np.argmin(distances, axis=1)
    closest_ids = color_matching_df.loc[closest_indices, "ID"].tolist()
    return closest_ids



def simpson_diversity_index(data: list):
    """
    calculate the simpson diversity index for a list of data
    """
    n = len(data)
    freq = np.array(list(Counter(data).values()))
    pi = freq / n
    return 1 - np.sum(pi ** 2)

def shannon_diversity_index(data: list):
    """
    calculate the shannon diversity index for a list of data
    """
    n = len(data)
    freq = np.array(list(Counter(data).values()))
    pi = freq / n
    return -np.sum(pi * np.log(pi))

def simpsons_diversity_index_distribution(data: list):
    return 1 - np.sum([p**2 for p in data])

def shannon_diversity_index_distribution(data: list):
    return -np.sum([p * np.log(p) for p in data])

def jointplot(df, name, save=True):
    """
    creates joinplot of the 2 dimensional color values
    """
    df = df.copy()

    if len(df) > 10000:
        df = df.sample(10000)

    joint_grid = sns.jointplot(
        x="pca_1", y="pca_2", data=df, kind="kde", colors="black", color="black"
    )

    df['rgb'] = df['median_lab_pollen_color'].apply(lambda x: color.lab2rgb(np.array(x).reshape(1, 1, 3)).reshape(3))
    colors = np.array([*df["rgb"].to_numpy() * 255]).astype(np.uint8).tolist()
    joint_grid.ax_joint.scatter(
        x=df["pca_1"].values,
        y=df["pca_2"].values,
        color=[tuple([x[0] / 255, x[1] / 255, x[2] / 255]) for x in colors],
    )

    if save:
        pass
        # plt.savefig(os.path.join(DATA_DEST, "on_device_plots", f"{name}_jointplot.png"))
    else:
        plt.show()
    plt.close()
    return joint_grid
