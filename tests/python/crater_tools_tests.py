import os, sys
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", ".."))

import numpy as np
import pandas as pd
from fcm import crater_tools


def test_effective_diameter():
    
    dummy_craters = np.array([[1, 2, 3], [10, 20, 30]])
    d_eff = crater_tools.effective_diameter(dummy_craters[0])
    np.testing.assert_allclose(d_eff, 6.6038545)
    
    d_eff_2 = crater_tools.effective_diameter(dummy_craters)
    np.testing.assert_allclose(d_eff_2, [d_eff, 10*d_eff])
    
    dummy_craters = pd.read_csv(os.path.join(THIS_DIR, "sample_cluster.csv"), index_col=0)
    d_eff = crater_tools.effective_diameter(dummy_craters.r.to_numpy())
    np.testing.assert_allclose(d_eff, 9.884990621782576)
    

def test_dispersion():
    
    dummy_craters = pd.DataFrame({"x": [-1, 1, -1, 1, 0], "y": [1, 1, -1, -1, 0]})
    disp = crater_tools.dispersion(dummy_craters)
    value_counts = pd.Series(disp).value_counts()
    
    expected = pd.Series({np.sqrt(2): 4, 2.0: 4, 2*np.sqrt(2): 2})
    pd.testing.assert_series_equal(value_counts, expected)
    
    dummy_craters = pd.read_csv(os.path.join(THIS_DIR, "sample_cluster.csv"), index_col=0)
    disp = crater_tools.dispersion(dummy_craters.iloc[:3])
    np.testing.assert_allclose(np.sort(disp), [13.06700178, 25.58476293, 36.78969209])
    

def test_ellipse():
    
    dummy_craters = pd.read_csv(os.path.join(THIS_DIR, "sample_cluster.csv"), index_col=0)
    aspect, center, components = crater_tools.ellipse(dummy_craters, bootstrap_samples=100,
                                                      seed=123)
    np.testing.assert_allclose(aspect, 0.3154624369511341)
    np.testing.assert_allclose(center, [0, 0], atol=1e-8)
    np.testing.assert_allclose(components,
                               [[-34.92610814, -88.91326344], [281.8505566, -11.01787519]])
    
    dummy_craters[['x', 'y']] = dummy_craters[['y', 'x']]
    aspect2, center2, components2 = crater_tools.ellipse(dummy_craters, bootstrap_samples=100,
                                                         seed=123)
    
    np.testing.assert_allclose(aspect2, aspect)
    np.testing.assert_allclose([center2[1], center2[0]], center)
    np.testing.assert_allclose(np.abs([components2[1], components2[0]]), np.abs(components))


def test_f_n_larger_d_max_frac():
    
    dummy_radii = np.array([[1, 3, 4, 1, 2], [4, 12, 16, 4, 8]])
    
    assert crater_tools.n_larger_D_max_frac(dummy_radii[0], 0.4) == 3
    np.testing.assert_allclose(crater_tools.f_larger_D_max_frac(dummy_radii[0], 0.4), 0.6) 
    np.testing.assert_allclose(crater_tools.n_larger_D_max_frac(dummy_radii, 0.4), [3, 3])
    np.testing.assert_allclose(crater_tools.f_larger_D_max_frac(dummy_radii, 0.4), [0.6, 0.6])
    
    assert crater_tools.n_larger_D_max_frac(dummy_radii[0], 0.6) == 2
    np.testing.assert_allclose(crater_tools.f_larger_D_max_frac(dummy_radii[0], 0.6), 0.4) 
    np.testing.assert_allclose(crater_tools.n_larger_D_max_frac(dummy_radii, 0.6), [2, 2])
    np.testing.assert_allclose(crater_tools.f_larger_D_max_frac(dummy_radii, 0.6), [0.4, 0.4])
    

def test_lon_lat_to_meters():
    
    dummy_craters = pd.read_csv(os.path.join(THIS_DIR, "sample_cluster.csv"), index_col=0)
    xy = dummy_craters[["x", "y"]].copy()
    
    lat_center = 30
    lon_center = 120
    Rp = 3e6
    
    dummy_craters["lat"] = dummy_craters["x"] * 180 / (Rp * np.pi) + lat_center
    dummy_craters["lon"] = dummy_craters["y"] * 180 / (Rp * np.pi * np.cos(lat_center * np.pi/180))\
        + lon_center
    
    crater_tools.lon_lat_to_meters(dummy_craters, Rp)
    pd.testing.assert_frame_equal(dummy_craters[["x", "y"]], xy)
        
    
def convergence_analysis_ellipse():
    
    import matplotlib.pyplot as plt
    dummy_craters = pd.read_csv(os.path.join(THIS_DIR, "sample_cluster.csv"), index_col=0)
    crater_tools.plot_craters(dummy_craters, ellipse_color="blue")
    plt.show()
    
    aspects, angles = list(), list()
    iterations = np.array([1, 2, 3, 5, 8, 10, 15, 25, 40, 50, 60, 70, 80, 90, 100, 120, 150,
                           250, 400, 600, 1000, 2000, 4000, 10000])
    for iters in iterations:
        aspect, _, components = crater_tools.ellipse(dummy_craters, bootstrap_samples=iters)
        aspects.append(aspect)
        angles.append(np.arctan2(components[1, 0], components[0, 0]))
    
    aspects = np.array(aspects)
    angles = np.array(angles)
    
    plt.loglog(iterations[:-1], np.abs(aspects[:-1] - aspects[-1]), label="aspect")
    plt.loglog(iterations[:-1], np.abs(angles[:-1] - angles[-1]), label="angle")
    plt.xlabel("boostrap samples")
    plt.ylabel("error")
    plt.title("angle = {:.3f}, aspect = {:.3f}".format(angles[-1], aspects[-1]))
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    test_effective_diameter()
    test_dispersion()
    test_ellipse()
    test_f_n_larger_d_max_frac()
    test_lon_lat_to_meters()
    convergence_analysis_ellipse()
