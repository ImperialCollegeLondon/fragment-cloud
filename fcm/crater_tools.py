import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from matplotlib.patches import Circle, Ellipse
from sklearn.decomposition import PCA
from sklearn.utils import resample


def plot_craters(craters, circle_color="black", ellipse_color=None, figsize=None):
    x_min = (craters.x - craters.r).min()
    x_max = (craters.x + craters.r).max()
    y_min = (craters.y - craters.r).min()
    y_max = (craters.y + craters.r).max()
    
    margin = 0.05
    x_min, x_max = x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min)
    y_min, y_max = y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_ylim(x_min, x_max)
    ax.set_xlim(y_min, y_max)
    
    for row in craters.itertuples():
        circle = Circle((row.y, row.x), row.r, color=circle_color, fill=False)
        ax.add_artist(circle)
    
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_aspect('equal')
    
    if ellipse_color is not None:
        _, center, components = ellipse(craters)
        ell = Ellipse((center[1], center[0]),
                      np.linalg.norm(components[0]), np.linalg.norm(components[1]),
                      180/np.pi * np.arctan2(components[0, 1], components[0, 0]),
                      color=ellipse_color, fill=False)
        ax.add_artist(ell)
    
    return fig


def effective_diameter(radii):
    
    return 2 * np.cbrt((radii**3).sum(axis=-1))


def ellipse(craters, bootstrap_samples=300, seed=6549810):

    if craters.shape[0] > 5:
        coordinates = craters[['x', 'y']].to_numpy()
        
        pca = PCA(2)
        explained_variances = list()
        pca_main_angles = list()
        
        if bootstrap_samples > 1:
            rs = np.random.RandomState(seed)
            for _ in range(bootstrap_samples):
                pca.fit(resample(coordinates, replace=True, random_state=rs))
                explained_variances.append(np.sqrt(pca.explained_variance_))
                pca_main_angles.append(np.arctan2(pca.components_[0, 0], pca.components_[0, 1]))
        else:
            pca.fit(coordinates)
            explained_variances.append(np.sqrt(pca.explained_variance_))
            pca_main_angles.append(np.arctan2(pca.components_[0, 0], pca.components_[0, 1]))
        
        center = coordinates.mean(axis=0)
        explained_variances = np.array(explained_variances)
        aspect = np.median(explained_variances[:, 1] / explained_variances[:, 0])
        
        angle = np.median(pca_main_angles)
        components = np.array([[np.cos(angle), -aspect * np.sin(angle)],
                               [np.sin(angle), aspect * np.cos(angle)]])
        components *= 3 * np.linalg.norm(explained_variances.mean(axis=0))
        
    else:
        assert craters.shape[0] > 0, "craters must not be empty"
        aspect = 1
        if craters.shape[0] == 1:
            center = craters[['x', 'y']].to_numpy().flatten()
            components = np.array([[craters.r[0], 0], [0, craters.r[0]]])
        else:
            center = craters[['x', 'y']].to_numpy().mean(axis=0)
            d_eff = effective_diameter(craters.r.to_numpy())
            components = np.array([[d_eff/2, 0], [0, d_eff/2]])        
        
    return aspect, center, components


def dispersion(craters):
    coordinates = craters[['x', 'y']].to_numpy()
    distances = pdist(coordinates)
    if distances.size == 0:
        distances = np.array(0.0)
    
    return distances
    

def n_larger_D_max_frac(radii, frac=0.25):
    
    r_max = radii.max()
    
    return (radii > r_max*frac).sum(axis=-1)
    

def f_larger_D_max_frac(radii, frac=0.25):
    
    r_max = radii.max()
    
    return (radii > r_max*frac).mean(axis=-1)


def lon_lat_to_meters(craters, Rp):
    
    lon_center, lat_center = craters.lon.mean(), craters.lat.mean()
    craters['x'] = Rp * (craters.lat - lat_center) * np.pi / 180
    craters['y'] = Rp * np.cos(lat_center * np.pi / 180) * (craters.lon - lon_center) * np.pi / 180
    