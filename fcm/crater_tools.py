import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression

from fcm._fcm_class import _check_number


###################################################
def plot_craters(craters, r_min=0., crater_color="black", ellipse_color=None, figsize=None,
                     px_lim=None, py_lim=None, fig=None, ax=None, flipxy=False, centre_mean=False):
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

    px_lim : tuple[float], optional
        x-limits of plot axis

    py_lim : tuple[float], optional
        y-limits of plot axis

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


    if flipxy:
        print("Renaming columns")
        craters = craters.rename(columns={"x": "y", "y": "x"})

    if centre_mean:
        craters.x = craters.x - craters.x.mean()
        craters.y = craters.y - craters.y.mean()
        
    x_min = (craters.x - craters.r).min()
    x_max = (craters.x + craters.r).max()
    y_min = (craters.y - craters.r).min()
    y_max = (craters.y + craters.r).max()
    
    margin = 0.05
    x_min, x_max = x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min)
    y_min, y_max = y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min)

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if px_lim is None:
        ax.set_xlim(x_min, x_max)
    else:
        ax.set_xlim(px_lim)
    if py_lim is None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(py_lim)
    
    for row in craters.itertuples():
        circle = Circle((row.x, row.y), row.r, color=crater_color, fill=False)
        if (row.r > r_min): ax.add_artist(circle)
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')

    bearing = 0.
    if ellipse_color is not None:
        _, center, components = ellipse(craters[craters.r > r_min])
        bearing = 180/np.pi * np.arctan2(components[1, 0], components[0, 0])
        ell = Ellipse((center[0], center[1]),
                      np.linalg.norm(components[:, 0]), np.linalg.norm(components[:, 1]),
                      180/np.pi * np.arctan2(components[1, 0], components[0, 0]),
                      color=ellipse_color, fill=False)
        ax.add_artist(ell)
    
    return fig, bearing


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


############################################################################
def ellipse_coleman(craters):
    '''calculates the best fit ellipse and aspect ratio
    the crater tools ellipse function cannot calculate aspect for clusters <6 craters >40 observations are 2-5 craters
    this function can calculate the ellipse for clusters of 2 craters or more'''
    
    x,y=craters['y'],craters['x']
    xy=craters[['y', 'x']]

    #take centre of ellipse as mean of coordinates of craters
    centerx=np.mean(x)
    centery=np.mean(y)

    #calculate angle to rotate the ellipse by rotating towards the crater furthest from the centre
    maxdist=0
    xpoint=0
    ypoint=0
    for a,b in zip(x,y):
        dist=np.sqrt((a-centerx)**2+(b-centery)**2)
        if dist>maxdist:
            maxdist=dist
            xpoint=a
            ypoint=b
    if((xpoint-centerx))!=0:
        rot=np.rad2deg(np.arctan((ypoint-centery)/(xpoint-centerx)))
    else:
        rot=0
    #retrieves the coordinates that make up the craters made using matplotlib.patches.Circle
    craterpointsx,craterpointsy=[],[]
    for row in craters.itertuples():
            circle = Circle((row.y, row.x), row.r, color="black", fill=False)
            path = circle.get_path()
            transform = circle.get_transform()
            newpath = transform.transform_path(path)
            circums=newpath.cleaned().iter_segments()
            for i in circums:
                craterpointsx.append(i[0][0])
                craterpointsy.append(i[0][1])
    craterpoints=list(zip(craterpointsx,craterpointsy))

    done=False
    donew=False
    doneh=False
    f=1 
    #choose starting width and height as the maximum and minimum of largest distance between the most distant x and y values 
    w=max(max(craterpointsx)-min(craterpointsx),max(craterpointsy)-min(craterpointsy))
    h=min(max(craterpointsx)-min(craterpointsx),max(craterpointsy)-min(craterpointsy))
    
    #begin fitting the ellipse
    while done==False:
        e=Ellipse(xy=(centerx,centery),width=w,height=h,angle=rot,fill=False,edgecolor='navajowhite',alpha=.4)
        #expand the ellipse width and height by 1% until it contains all the craters completely within it
        if e.contains_points(craterpoints,radius=0).all()==True:

            wsf=1
            while donew==False:
                e=Ellipse(xy=(centerx,centery),width=w,height=h,angle=rot,fill=False,edgecolor='plum',alpha=.4)
                #attempt to shrink the ellipse in the width direction until the craters are no longer contained
                #then revert back one shrinking step so the craters leave the loop contained
                if e.contains_points(craterpoints,radius=0).all()==False:
                    w=w/wsf
                    donew=True
                else:
                    wsf=0.99#wsf-fi
                    w=w*wsf

            hsf=1        
            while doneh==False:
                e=Ellipse(xy=(centerx,centery),width=w,height=h,angle=rot,fill=False,edgecolor='aqua',alpha=.4)
                #attempt to shrink the ellipse in the width direction until the craters are no longer contained
                #then revert back one shrinking step so the craters leave the loop contained
                if e.contains_points(craterpoints,radius=0).all()==False:
                    h=h/hsf
                    doneh=True
                else:
                    hsf=0.99#hsf-fi
                    h=h*hsf
            
            #plot the final best fit ellipse and calculate aspect,eccentricity(unused),area of the ellipse(unused)
            e=Ellipse(xy=(centerx,centery),width=w,height=h,angle=rot,fill=False,edgecolor='springgreen')
            aspect=abs(min(w,h)/max(w,h))
            ecc=np.sqrt(np.square(max(w,h)/2)-np.square(min(w,h)/2))/max(w,h)
            area=np.pi*w*h
            done=True

        else:
            f=1.01#f+fi
            w=w*f
            h=h*f

    #fig,ax=plt.subplots()     ### comment out this line and below if plotting not wanted
    #showellipse(craters,ax)	###plots the ellipse around the craters

    #return aspect, eccentricity, centre coordinates,major axis length, minor axis length, rotation of ellipse, area of ellispe
    return (aspect,ecc,e.get_center(),max(w,h),min(w,h),rot,area)



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
def csfd_fit(craters, min_craters = 5):
    """Fit a power law through the cumulative size-frequency distribution of craters
       in the cluster and return the exponent and max diameter.

    Parameters
    ----------
    craters :  pandas.DataFrame

    min_craters : int
         minimum number of craters in cluster to fit CSFD

    """

    d = 2*craters['r'].to_numpy()
    d_max = d.max()
    d_min = d.min()
    d_med = np.median(d)
    
    if len(craters) < min_craters:
        d_exp = 0.
    else:
        cfd = craters['r'].value_counts().sort_index()[::-1].cumsum()
        X = np.log10(cfd.index.values.reshape(-1,1))
        Y = np.log10(cfd.values.reshape(-1,1))
        linear_regressor = LinearRegression()  # create object for the class
        sol = linear_regressor.fit(X, Y)  # perform linear regression
        d_exp = sol.coef_[0][0]

    return d_exp, d_max, d_med, d_min

    
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

##############################################################
def cluster_characteristics(craters, r_min=0.):
    """Calculate a suite of characteristics for a given cluster.
    
    Parameters
    ----------
    craters : pandas.DataFrame
        list of craters in cluster
        columns = ['x', 'y', 'r']
        units must be the same for all columns, typically meters

    Returns
    -------
    cdict : Dictionary
        a dictionary of cluster characteristics with the following entries:
        * Effective diameter [m] -- (Sum D_i**3)**0.333
        * No. of craters -- total number of craters in the cluster (larger than minimum size)
        * Dispersion [m] -- median distance between crater pairs
        * Aspect ratio -- width/length of best-fitting ellipse around cluster
        * No. of large craters -- No. of craters with D > f*Dmax
        * Large crater fraction -- Proportion of craters with D > f*Dmax
    """

    # Populate the dictionary with default values for no craters
    cdict = {'Effective Diameter [m]': 0., 
             'No. of Craters': 0,
             'Dispersion [m]': 0.,
             'Aspect Ratio': 0.,
             'Aspect Ratio (alt)': 1.,
             'Large Crater Fraction': 0.,
             'CSFD exponent': 0.,
             'Maximum Diameter [m]': 0.,
             'Median Diameter [m]': 0.,
             'Minimum Diameter [m]': 0.,
             'No. of Large Craters': 0}

    if craters is None:
        return cdict

    # Filter out small craters if necessary
    craters = craters[craters['r'] > r_min]
    radii = craters['r'].to_numpy()
    
    # If no craters left, treat as airburst
    if len(radii) == 0:
        return cdict
    
    # Compute cluster characteristics if more than one crater
    if len(radii) > 1:
        cdict['Effective Diameter [m]'] = effective_diameter(radii)
        cdict['No. of Craters'] = len(craters)
        cdict['Dispersion [m]'] = np.median(dispersion(craters))
        ellipse_params = ellipse_coleman(craters)
        cdict['Aspect Ratio'] = ellipse_params[0]
        ellipse_params = ellipse(craters)
        cdict['Aspect Ratio (alt)'] = ellipse_params[0]
        csfd_exp, d_max, d_med, d_min = csfd_fit(craters) 
        cdict['CSFD exponent'] = csfd_exp
        cdict['Maximum Diameter [m]'] = d_max
        cdict['Median Diameter [m]'] = d_med
        cdict['Minimum Diameter [m]'] = d_min
        cdict['No. of Large Craters'] = n_larger_D_max_frac(radii, frac=0.5)
        cdict['Large Crater Fraction'] = f_larger_D_max_frac(radii, frac=0.5)

    # Otherwise define values for single crater
    else:
        cdict = {'Effective Diameter [m]': 2*radii[0], 
                 'No. of Craters': 1,
                 'Dispersion [m]': 0.,
                 'Aspect Ratio': 1.,
                 'Aspect Ratio (alt)': 1.,
                 'Large Crater Fraction': 1.,
                 'CSFD exponent': 0.,
                 'Maximum Diameter [m]': 2*radii[0],
                 'Median Diameter [m]': 2*radii[0],
                 'Minimum Diameter [m]': 2*radii[0],
                 'No. of Large Craters': 1}
            
    return cdict


##############################################################
def fragments_characteristics(fragments):
    """Calculate a suite of characteristics for the population of
       fragments at the end of a run.
    
    Parameters
    ----------
    fragments : pandas.DataFrame
        list of fragments in population

    Returns
    -------
    fdict : Dictionary
        a dictionary of fragment populations characteristics
    """

    # Populate the dictionary with default values for no craters
    fdict = {'No. of fragments': 0,
             'Total mass [kg]': 0.,
             'Total momentum [Ns]': 0.,
             'Total vert. momentum [Ns]': 0.,
             'Maximum mass [kg]': 0.,
             'Median mass [kg]': 0.,
             'Minimum mass [kg]': 0.,
             'Mean mass [kg]': 0.,
             'Std. dev. mass [kg]': 0.,
             'Maximum velocity [km/s]': 0.,
             'Median velocity [km/s]': 0.,
             'Minimum velocity [km/s]': 0.,
             'Mean velocity [km/s]': 0.,
             'Std. dev. velocity [km/s]': 0.,
             'Maximum angle': 0.,
             'Median angle': 0.,
             'Minimum angle': 0.,
             'Mean angle': 0.,
             'Std. dev. angle': 0.,
             'Maximum time [s]': 0.,
             'Median time [s]': 0.,
             'Mean time [s]': 0.,
             'Std. dev. time [s]': 0.,
             'Maximum momentum [Ns]': 0.,
             'Median momentum [Ns]': 0.,
             'Minimum momentum [Ns]': 0.,
             'Mean momentum [Ns]': 0.,
             'Std. dev. momentum [Ns]': 0.,
             'Maximum vert. momentum [Ns]': 0.,
             'Median vert. momentum [Ns]': 0.,
             'Minimum vert. momentum [Ns]': 0.,
             'Mean vert. momentum [Ns]': 0.,
             'Std. dev. vert. momentum [Ns]': 0.}

    if fragments is None:
        return fdict

    # Fragments of interest are those hitting the ground
    f = fragments[fragments.h < 1.E-6]

    # Return blank dictionary if nothing hits the ground
    if len(f) == 0:
        return fdict
    
    # Determine separation time of impact
    f['dt'] = f.t - f.t.min()
    f['mv'] = f.m * f.v * 1000.
    f['mvv'] = f.mv * np.sin(np.radians(f.angle))
    fstats = f.describe()

    # Populate the dictionary with default values for no craters
    fdict['No. of fragments']             = len(f)
    fdict['Total mass [kg]']              = f.m.sum()
    fdict['Total momentum [Ns]']          = f.mv.sum()
    fdict['Total vert. momentum [Ns]']    = f.mvv.sum()
    fdict['Maximum mass [kg]']            = fstats.m.loc['max']
    fdict['Median mass [kg]']             = fstats.m.loc['50%']
    fdict['Minimum mass [kg]']            = fstats.m.loc['min']
    fdict['Mean mass [kg]']               = fstats.m.loc['mean']
    fdict['Maximum velocity [km/s]']      = fstats.v.loc['max']
    fdict['Median velocity [km/s]']       = fstats.v.loc['50%']
    fdict['Minimum velocity [km/s]']      = fstats.v.loc['min']
    fdict['Mean velocity [km/s]']         = fstats.v.loc['mean']
    fdict['Maximum angle']                = fstats.angle.loc['max']
    fdict['Median angle']                 = fstats.angle.loc['50%']
    fdict['Minimum angle']                = fstats.angle.loc['min']
    fdict['Mean angle']                   = fstats.angle.loc['mean']
    fdict['Maximum time [s]']             = fstats.dt.loc['max']
    fdict['Median time [s]']              = fstats.dt.loc['50%']
    fdict['Mean time [s]']                = fstats.dt.loc['mean']
    fdict['Maximum momentum [Ns]']        = fstats.mv.loc['max']
    fdict['Median momentum [Ns]']         = fstats.mv.loc['50%']
    fdict['Minimum momentum [Ns]']        = fstats.mv.loc['min']
    fdict['Mean momentum [Ns]']           = fstats.mv.loc['mean']
    fdict['Maximum vert. momentum [Ns]']  = fstats.mvv.loc['max']
    fdict['Median vert. momentum [Ns]']   = fstats.mvv.loc['50%']
    fdict['Minimum vert. momentum [Ns]']  = fstats.mvv.loc['min']
    fdict['Mean vert. momentum [Ns]']     = fstats.mvv.loc['mean']

    if len(f) > 1:
        fdict['Std. dev. mass [kg]']           = fstats.m.loc['std']
        fdict['Std. dev. velocity [km/s]']     = fstats.v.loc['std']
        fdict['Std. dev. angle']               = fstats.angle.loc['std']
        fdict['Std. dev. time [s]']            = fstats.dt.loc['std']
        fdict['Std. dev. momentum [Ns]']       = fstats.mv.loc['std']
        fdict['Std. dev. vert. momentum [Ns]'] = fstats.mvv.loc['std']
        
    return fdict
