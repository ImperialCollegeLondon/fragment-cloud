import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.utils import resample

from fcm._fcm_class import _check_number


###################################################
def plot_craters(craters, crater_color="black", ellipse_color=None, figsize=None):
    """Make a matplotlib plot of a crater cluster.
    Optionally computes and plots best-fit ellipse around cluster
    
    Parameters
    ----------
    craters : pandas.DataFrame
        list of craters in cluster
        columns = ['x', 'y', 'r']
        units must be the same for all columns, typically meters
    
    crater_color : str, optional
        color of circles representing craters
            a Matplotlib color, see
            https://matplotlib.org/gallery/color/named_colors.html?highlight=list%20named%20colors
        default: black
    
    ellipse_color : str, optional
        color of best-fit ellipse; a Matplotlib color
        If specified, best-fit ellipse is computed and drawn
    
    figsize : tuple[int], optional
        size of figure in inches

    Returns
    -------
    matplotlib.figure.Figure

    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    assert isinstance(craters, pd.DataFrame),\
        "craters must be a pandas.DataFrame instance"
    assert pd.Index(["x", "y", "r"]).isin(craters.columns).all(),\
        "craters must have columns 'x', 'y', and 'r'"
    x_min = (craters.x - craters.r).min()
    x_max = (craters.x + craters.r).max()
    y_min = (craters.y - craters.r).min()
    y_max = (craters.y + craters.r).max()
    
    margin = 0.05
    x_min, x_max = x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min)
    y_min, y_max = y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    for row in craters.itertuples():
        circle = Circle((row.x, row.y), row.r, color=crater_color, fill=False)
        ax.add_artist(circle)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    
    if ellipse_color is not None:
        _, center, components = ellipse(craters)
        ell = Ellipse((center[0], center[1]),
                      np.linalg.norm(components[:, 0]), np.linalg.norm(components[:, 1]),
                      180/np.pi * np.arctan2(components[1, 0], components[0, 0]),
                      color=ellipse_color, fill=False)
        ax.add_artist(ell)
    
    return fig


###################################################
def effective_diameter(radii):
    """Calculate effective diameter of a crater cluster
    
    d_eff = (sum of radii^3)^1/3
    
    Parameters
    ----------
    radii : numpy.ndarray
        numpy array with radii of craters in the cluster
        If array is more than 1-dimensional, computes effective diameter over last axis
    
    Returns
    -------
    Union[float, numpy.ndarray]
        If input is 1-D, just the effective diameter (float)
        If input is n-D, (numpy.ndarray) with n-1 dimensions; data = effective radii
    """
    
    return 2 * np.cbrt((radii**3).sum(axis=-1))


###################################################
def ellipse(craters, min_craters=5, bootstrap_samples=80, seed=74527):
    """Calculate best-fit ellipse around crater cluster with PCA,
    if cluster has at least min_craters.
    Otherwise, just places a circle in the centre with diameter = effective diameter.
    Performs bootstrapping to minimize influence of outliers.
    
    Parameters
    ----------
    craters : pandas.DataFrame
        has columns [x, y, r]
        list of x, y coordiantes and radii of craters in cluster
    
    min_craters : int, optional
        minimum number of craters in a cluster where ellipse is calculated
        min_craters >= 2
        default = 5
    
    bootstrap_samples : int, optional
        number of repetitions with bootstrapped samples from crater cluster
        bootstrap_samples >= 1
        default = 80
    
    seed : int, optional
        seed for the random number generator used for bootstrapping
        
    Returns
    -------
    aspect : float
        aspect ratio of best-fit ellipse
        1 = circle, 0 = line
        eccentricity = sqrt(1 - aspect^2)
    
    center : numpy.array([x, y])
        coordinates of ellipse center
    
    components : np.array([x1, x2], [y1, y2])
        components of semi-major (1) and semi-minor (2) axis
    """
    min_craters = _check_number(min_craters, "min_craters", False, 2, True, bound_format="{:d}")
    bootstrap_samples = _check_number(bootstrap_samples, "bootstrap_samples", False, 1, True,
                                      bound_format="{:d}")
    seed = _check_number(seed, "seed", False, 0, True, bound_format="{:d}")
    
    if craters.shape[0] >= min_craters:
        coordinates = craters[['x', 'y']].to_numpy()
        
        pca = PCA(2)
        explained_variances = list()
        pca_main_angles = list()
        
        if bootstrap_samples > 1:
            rs = np.random.RandomState(seed)
            for _ in range(bootstrap_samples):
                pca.fit(resample(coordinates, replace=True, random_state=rs))
                explained_variances.append(np.sqrt(pca.explained_variance_))
                pca_main_angles.append(np.arctan2(pca.components_[0, 1], pca.components_[0, 0]))
        else:
            pca.fit(coordinates)
            explained_variances.append(np.sqrt(pca.explained_variance_))
            pca_main_angles.append(np.arctan2(pca.components_[0, 1], pca.components_[0, 0]))
        
        center = coordinates.mean(axis=0)
        explained_variances = np.array(explained_variances)
        # calculate median aspect ratio
        aspect = np.median(explained_variances[:, 1] / explained_variances[:, 0])
        
        # bring all angles into same half of the circle (around 0 or pi/2, depending)
        pca_main_angles = np.array(pca_main_angles)
        if ((np.pi/4 < np.abs(pca_main_angles)) & (3*np.pi/4 > np.abs(pca_main_angles))).mean() > 0.5:
            pca_main_angles[pca_main_angles < 0] += np.pi
        else:
            pca_main_angles[pca_main_angles > np.pi/2] -= np.pi
            pca_main_angles[pca_main_angles < -np.pi/2] += np.pi
        
        # calcualte median angle of main component
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


###################################################
def dispersion(craters):
    """Calculate pairwise distane of crater centers
    
    Parameters
    ----------
    craters : pandas.DataFrame
        has columns [x, y]
        list of x, y coordiantes of craters in a cluster
    
    Returns
    -------
    numpy.ndarray
        1D numpy array with pairwise distances, same unit as input units
    """
    coordinates = craters[['x', 'y']].to_numpy()
    distances = pdist(coordinates)
    if distances.size == 0:
        distances = np.array(0.0)
    
    return distances
    

###################################################
def n_larger_D_max_frac(radii, frac=0.25):
    """Count the number of craters in a cluster that are larger than
    a fraction of the size of the largest crater in the cluster
    
    Parameters
    ----------
    radii : numpy.ndarray
        array of radii of craters in cluster
        If array is more than 1-dimensional, counts over last axis
    
    frac : float, optional
        size fraction above which crater is counted
        0 <= frac <= 1
        default = 0.25
    
    Returns
    -------
    Union[int, numpy.ndarray]
        If input is 1D, just the number of large craters
        If input is nD, (numpy.ndarray) with n-1 dimensions; data = number of large craters
    """
    frac = _check_number(frac, "frac", True, 0, True, 1, True)
    r_max = radii.max(axis=-1)
    comp = np.repeat(r_max.reshape(radii.shape[:-1] + (1,)), radii.shape[-1], -1)
    
    return (radii > comp*frac).sum(axis=-1)
    

###################################################
def f_larger_D_max_frac(radii, frac=0.25):
    """Compute the fraction of craters in a cluster that are larger than
    a fraction of the size of the largest crater in the cluster
    
    Parameters
    ----------
    radii : numpy.ndarray
        array of radii of craters in cluster
        If array is more than 1-dimensional, computes over last axis
    
    frac : float, optional
        size fraction above which crater is counted
        0 <= frac <= 1
        default = 0.25
    
    Returns
    -------
    Union[float, numpy.ndarray]
        If input is 1D, just the number of large craters
        If input is nD, (numpy.ndarray) with n-1 dimensions; data = number of large craters
    """
    frac = _check_number(frac, "frac", True, 0, True, 1, True)
    r_max = radii.max(axis=-1)
    comp = np.repeat(r_max.reshape(radii.shape[:-1] + (1,)), radii.shape[-1], -1)
    
    return (radii > comp*frac).mean(axis=-1)


###################################################
def lon_lat_to_meters(craters, Rp):
    """Convert crater cluster with crater coordinates in (latitude, longitude), in degrees,
    to meters distance from the cluster center
    
    Adds/overwrites columns [x, y] in the craters parameter
    
    Parameters
    ----------
    craters : pandas.DataFrame
        has columns [lat, lon]
        latitude and longitude of craters in a cluster, in degrees
        
    Rp : float
        radius of the planet, in meters
        Rp > 0
    """
    Rp = _check_number(Rp, "Rp", True, 0)
    if not pd.Index(["lat", "lon"]).isin(craters.columns).all():
        raise KeyError("craters must have columns lat ('latitude') and lon ('longitude')")
    
    lon_center, lat_center = craters.lon.mean(), craters.lat.mean()
    craters['x'] = Rp * (craters.lat - lat_center) * np.pi / 180
    craters['y'] = Rp * np.cos(lat_center * np.pi / 180) * (craters.lon - lon_center) * np.pi / 180
    