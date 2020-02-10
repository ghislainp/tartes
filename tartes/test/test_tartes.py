
from tartes import albedo
import numpy as np


def test_one_layer_diffuse():
    ssa = 20      # in m^2/kg
    wavelength = [450e-9, 650e-9, 850e-9, 1030e-9, 1300e-9]  # in m

    alb = albedo(wavelength, ssa)
    
#   w2008 res = np.array((0.99614203926738309, 0.96047324904990949, 0.88128514422618554, 0.66596004345176796, 0.42830398442634676))
    res = np.array([0.98987212, 0.96047325, 0.88128514, 0.66596004, 0.42830398])
    assert np.all(abs(alb - res) < 1e-8)


def test_w1995():
    ssa = 20
    wavelength = 550e-9

    alb = albedo(wavelength, ssa, refrac_index="w1995")
    assert abs(alb - 0.9798191182503498) < 1e-10


def test_one_layer_direct():

    ssa = 20
    wavelength = 850e-9

    alb = albedo(wavelength, ssa, dir_frac=1, sza=30)

    assert abs(alb - 0.8583967584060925) < 1e-10


def get_multilayer():

    ssa = [20, 15, 10]
    density = [200, 250, 300]
    thickness = [0.01, 0.10, 1000]

    return ssa, density, thickness


def test_multilayer():

    ssa, density, thickness = get_multilayer()

    wavelength = [450e-9, 650e-9, 850e-9, 1030e-9, 1300e-9]  # in m

    alb = albedo(wavelength, ssa, density, thickness)

    res = np.array([0.98697191, 0.95453273, 0.87187588, 0.65985864, 0.42739316])
    assert np.all(abs(alb - res) < 1e-8)


def test_diffus():

    ssa, density, thickness = get_multilayer()
    wavelength = np.arange(400, 1000, 200) * 1e-9

    alb_dir53 = albedo(wavelength, ssa, density, thickness, sza=53, dir_frac=1)
    alb_diff = albedo(wavelength, ssa, density, thickness, dir_frac=0)

    assert np.all(abs(alb_dir53 - alb_diff) < 1e-8)

    alb_halfdir53 = albedo(wavelength, ssa, density, thickness, sza=53, dir_frac=0.5)
    
    assert np.all(abs(alb_halfdir53 - alb_diff) < 1e-8)


def test_impurities():
    ssa = 20      # in m^2/kg
    wavelength = [450e-9, 650e-9, 850e-9, 1030e-9, 1300e-9]  # in m

    alb = albedo(wavelength, ssa, impurities=100e-9)
    
    print(alb)
    res = np.array([0.95566775, 0.94667078, 0.87767193, 0.66526393, 0.42814038])
    assert np.all(abs(alb - res) < 1e-8)