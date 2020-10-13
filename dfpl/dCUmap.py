import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

from dfpl import fingerprint as fp

# %matplotlib inline

sns.set(style='white', context='notebook', rc={'figure.figsize': (10, 14)})

# get penguin data for testing
penguins = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8"
                       "/data/penguins_size.csv")
penguins = penguins.dropna()
penguins.species_short.value_counts()
penguins.island.value_counts()

sns.pairplot(penguins, hue='species_short')
plt.show()

reducer = umap.UMAP()

# clean up
# no NAs, only measurement columsn
penguin_data = penguins[
    [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g"]
].values
# convert each feature into zscores since they are on different scales
scaled_penguin_data = StandardScaler().fit_transform(penguin_data)

# embed data into two-dim space
embedding = reducer.fit_transform(scaled_penguin_data)
embedding.shape
# --> each row in the original df retreivew 2D coordinates

# visualize this
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in penguins.species_short.map({
        "Adelie": 0, "Chinstrap": 1, "Gentoo": 2
    })]
)
plt.gca().set_aspect('equal', 'datalim')
plt.title("UMAP projection of the Penguin dataset", fontsize=24)
plt.show()

# lets try with some chemical fingerprints

df = fp.importDataFile("data/S_dataset_extended.pkl")
df2 = np.array(df[df['fp'].notnull()]['fp'].to_list())
er = df[df['fp'].notnull()]['ER'].fillna(-1)

fit = umap.UMAP(metric="jaccard")
%time u = fit.fit_transform(df2)

plt.title("UMAP projection of Sun et al dataset using jaccard metric")
plt.scatter(u[:,0], u[:,1],
            c=[sns.color_palette()[x] for x in er.map({-1.0:0, 0.0:1, 1.0:2})])
# plt.legend(loc='upper right')
plt.show()

df_d = fp.importDataFile("data/dsstox_20160701.pkl")
df_d2 = np.array(df_d[df_d['fp'].notnull()]['fp'].to_list())

fit_d = umap.UMAP(metric="jaccard")
%time u_d = fit_d.fit_transform(df_d2)

plt.title("UMAP projection of Sun et al dataset using jaccard metric")
plt.scatter(u[:,0], u[:,1])
# plt.legend(loc='upper right')
plt.show()
