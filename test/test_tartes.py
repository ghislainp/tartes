from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from tartes import albedo, actinic_profile, absorption_profile, irradiance_profiles

import pytest

# define the snowpack fixture


@dataclass
class Snowpack(object):
    SSA: npt.ArrayLike  # in m^2/kg
    density: npt.ArrayLike
    thickness: npt.ArrayLike

    def as_dict(self):
        return {'SSA': self.SSA, 'density': self.density, 'thickness': self.thickness}


@pytest.fixture
def snowpack_1layer_ssa20():
    return Snowpack(SSA=20, density=350, thickness=[10000])


@pytest.fixture
def snowpack_multilayer():
    return Snowpack(SSA=[20, 15, 10], density=[200, 250, 300], thickness=[0.01, 0.10, 1000])


def test_one_layer_diffuse(snowpack_1layer_ssa20):
    wavelength = [450e-9, 650e-9, 850e-9, 1030e-9, 1300e-9]  # in m

    alb = albedo(wavelength, **snowpack_1layer_ssa20.as_dict(), B0=1.6, g0=0.86, shape_parameterization='linear')

    print(alb)
    res = np.array([0.98935712, 0.9584951, 0.87560886, 0.6522192, 0.40995579])
    # res = np.array([0.98933755, 0.95842003, 0.87539395, 0.65170265, 0.40927269]) # change of diffuse angle
    #   w2008 res = np.array((0.99614203926738309, 0.96047324904990949, 0.88128514422618554, 0.66596004345176796, 0.42830398442634676))
    # res = np.array([0.98987212, 0.96047325, 0.88128514, 0.66596004, 0.42830398])
    assert np.allclose(alb, res)


def test_w1995(snowpack_1layer_ssa20):
    wavelength = 550e-9

    alb = albedo(
        wavelength,
        **snowpack_1layer_ssa20.as_dict(),
        refrac_index='w1995',
        B0=1.6,
        g0=0.86,
        shape_parameterization='linear',
    )
    print(alb)
    assert np.allclose(alb, 0.9787984375800363)
    # assert abs(alb - 0.9787596808617396) < 1e-10  # change of diffuse angle


def test_one_layer_direct(snowpack_1layer_ssa20):
    wavelength = 850e-9

    alb = albedo(
        wavelength,
        **snowpack_1layer_ssa20.as_dict(),
        dir_frac=1,
        sza=30,
        B0=1.6,
        g0=0.86,
        shape_parameterization='linear',
    )
    print(alb)

    assert abs(alb - 0.8583967584060925) < 1e-10


def test_multilayer(snowpack_multilayer):
    wavelength = [450e-9, 650e-9, 850e-9, 1030e-9, 1300e-9]  # in m

    alb = albedo(
        wavelength,
        SSA=snowpack_multilayer.SSA,
        density=snowpack_multilayer.density,
        thickness=snowpack_multilayer.thickness,
        B0=1.6,
        g0=0.86,
        shape_parameterization='linear',
    )

    res = np.array([0.98630862, 0.95225066, 0.86571522, 0.64578293, 0.40898261])  # valid for 48.18...°
    # res = np.array([0.98628342, 0.95216404, 0.8654819, 0.64525358, 0.40829709])  # valid for 48°
    print(alb)
    # res = np.array([0.98697191, 0.95453273, 0.87187588, 0.65985864, 0.42739316])  # valid for 53°
    assert np.allclose(alb, res)


def test_diffus(snowpack_multilayer):
    wavelength = np.arange(400, 1000, 200) * 1e-9

    sza_eff = np.rad2deg(np.arccos((7 / 3 - 1) / 2))

    alb_dir48 = albedo(
        wavelength, **snowpack_multilayer.as_dict(), sza=sza_eff, dir_frac=1, shape_parameterization='linear'
    )
    alb_diff = albedo(wavelength, **snowpack_multilayer.as_dict(), dir_frac=0, shape_parameterization='linear')

    assert np.allclose(alb_dir48, alb_diff)

    alb_halfdir48 = albedo(
        wavelength, **snowpack_multilayer.as_dict(), sza=sza_eff, dir_frac=0.5, shape_parameterization='linear'
    )

    assert np.allclose(alb_halfdir48, alb_diff)


def test_impurities(snowpack_1layer_ssa20):
    wavelength = [450e-9, 650e-9, 850e-9, 1030e-9, 1300e-9]  # in m

    alb = albedo(
        wavelength,
        **snowpack_1layer_ssa20.as_dict(),
        impurities=100e-9,
        B0=1.6,
        g0=0.86,
        shape_parameterization='linear',
    )

    print(alb)
    res = np.array([0.94342398, 0.939721, 0.87067581, 0.65127868, 0.40973576])  # valid for 48.18....°
    # res = np.array([0.95337112, 0.9439218, 0.8716147, 0.65098498, 0.40910791])  # valid for 48°
    # res = np.array([0.95566775, 0.94667078, 0.87767193, 0.66526393, 0.42814038])  # valid for 53°
    assert np.allclose(alb, res, atol=1e-8)


@pytest.mark.parametrize('dir_frac', [0.5, 1])
def test_flux_actinic(snowpack_1layer_ssa20, dir_frac):
    wavelength = 450e-9

    z = [0.1, 0.2]

    act = actinic_profile(
        wavelength,
        z,
        **snowpack_1layer_ssa20.as_dict(),
        dir_frac=dir_frac,
        sza=30,
        B0=1.6,
        g0=0.86,
        shape_parameterization='linear',
    )

    print(act)
    if dir_frac == 1:
        assert np.allclose(act, [3.03490924, 2.01385722])
    elif dir_frac == 0.5:
        assert np.allclose(act, [2.83752223, 1.88287826])
    # assert np.allclose(act, [3.21803673, 2.26422331], atol=1e-8)


def test_consistency_absorption(snowpack_1layer_ssa20):
    nlyr = 100

    wls = np.arange(400, 2000, 5) * 1e-9
    thickness = [1e-2] * nlyr + [1000]

    sza = 60

    sp = snowpack_1layer_ssa20.as_dict()
    sp['thickness'] = thickness

    alb = albedo(wls, **sp, dir_frac=1, sza=sza)
    z, absorption = absorption_profile(wls, **sp, dir_frac=1, sza=sza)

    assert np.allclose(alb, 1 - absorption.sum(axis=-1))


@pytest.mark.parametrize('dir_frac', [0, 0.5, 1])
def test_consistency_irradiance(snowpack_1layer_ssa20, dir_frac):
    nlyr = 100

    wls = np.arange(400, 2000, 5) * 1e-9
    thickness = [1e-2] * nlyr + [1000]

    sza = 60

    sp = snowpack_1layer_ssa20.as_dict()
    sp['thickness'] = thickness

    alb = albedo(wls, **sp, dir_frac=dir_frac, sza=sza)
    irr_dn, irr_up = irradiance_profiles(wls, z=[0], **sp, dir_frac=dir_frac, sza=sza)
    assert np.allclose(alb, irr_up / irr_dn)
