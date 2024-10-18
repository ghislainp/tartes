# -*-coding:utf-8 -*
#
# TARTES, Copyright (c) 2014-2023, Quentin Libois, Picard Ghislain
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#

from typing import Sequence, Union, Optional, Type, Generator
from collections.abc import Callable
import datetime

import numpy as np
import numpy.typing as npt

from scipy.linalg import solve_banded
import scipy.integrate

from .impurities import Impurities, SootSNICAR3
from .refractive_index import refice1995, refice2008, refice2016

# constants for convenience
C_SPEED = 299792458.
H_PLANCK = 6.62607015e-34 


# Many tartes arguments can be array or scalar of float/int
ArrayOrScalar = Union[npt.NDArray[np.floating], float, int]
FlexibleArray = Union[npt.NDArray[np.floating], float, int, Sequence[float]]


def ssa(r_opt: ArrayOrScalar) -> ArrayOrScalar:
    """return the specific surface area (SSA) from optical radius.

    :param r_opt: optical radius (m)
    :type r_opt: scalar or array

    :returns: SSA in m^2/kg"""
    return (3 / 917.0) / r_opt


default_B0 = 1.6  # Based on Libois et al. 2014 # only use with shape_parameterization="constant" and "linear"
robledano23_g0 = (0.64 + 1) / 2  # (=0.82) Middle range of Robledano et al. 2023

default_g0 = robledano23_g0

# deduce for the b value derived from Gallet et al. 2009 SSA versus albedo measurements.
# default_g0 = 0.86

default_y0 = 0.728
default_W0 = 0.0611

# B0 and g0 are related to b factor in Kokhanosky and Zege 2004 or Picard et al. 2009 by:
# b = 4/3*sqrt(B0/(1-g0))


def broadband_albedo(
    wavelength: FlexibleArray,
    totflux: FlexibleArray,
    dir_frac: FlexibleArray,
    SSA: FlexibleArray,
    **kwargs,
):
    """compute the broadband albedo of a snowpack specified by the profiles of SSA and density using TARTES. Refer to
    the :py:func:`albedo` function for a detailed description of the parameters.

    To compute the totflux and dir_frac for a simple atmosphere, see  :py:func:`atmospheric_incident_spectrum` function
    provided in this package.

    :param wavelength: wavelength (m)
    :type wavelength: array
    :param totflux: total spectral incident flux across the horizontal surface (direct+diffuse). 
        The direct flux can be calculated from the direct beam radiance F0 incoming at the incidence theta_inc with F0 x
        cos(theta_inc). The unit of totflux is at the user's discretion as TARTES internal calculation is relative to
        the incoming flux. :param dir_frac: spectrum of fraction of the direct radiation
    :param SSA: surface specific area (m^2/kg) for each layer
    :type SSA: array
    :param **kwagrs: all the other optional parametres of the :py:func:`albedo` function

    :returns: broadband albedo
    """

    kwargs['return_dir_diff'] = True
    dir_frac_np = np.atleast_1d(dir_frac)
    totflux = np.atleast_1d(totflux)

    albedo_dir, albedo_diff = albedo(wavelength, SSA, dir_frac=dir_frac, **kwargs)

    broadband_albedo = scipy.integrate.simpson(
        totflux * (dir_frac_np * albedo_dir + (1.0 - dir_frac_np) * albedo_diff), x=wavelength
    ) / scipy.integrate.simpson(totflux, x=wavelength)

    return broadband_albedo


def albedo(
    wavelength: ArrayOrScalar,
    SSA: ArrayOrScalar,
    density: Optional[ArrayOrScalar] = None,
    thickness: Optional[ArrayOrScalar] = None,
    shape_parameterization: str = 'robledano23',
    g0: ArrayOrScalar = default_g0,
    B0: Union[str, ArrayOrScalar] = default_B0,
    impurities: ArrayOrScalar = 0.0,
    impurities_type: Type[Impurities] = SootSNICAR3,
    refrac_index: Union[str, ArrayOrScalar, Callable] = 'p2016',
    soilalbedo: ArrayOrScalar = 0.0,
    dir_frac: ArrayOrScalar = 0.0,
    diff_method: str = 'aart eq',
    infprop_method: str = 'delta_eddington',
    sza: ArrayOrScalar = 0.0,
    return_dir_diff: bool = False,
) -> Union[ArrayOrScalar, tuple[ArrayOrScalar, ArrayOrScalar]]:
    """
    compute the spectral albedo of a snowpack specified by the profiles of SSA and density using TARTES. The underlying
    interface has an albedo specified by soilalbedo (0.0 by default). For semi-infinite snowpack, use thickness = None
    (the default value).

    :param wavelength: wavelength (m)
    :type wavelength: array
    :param SSA: surface specific area (m^2/kg) for each layer
    :type SSA: array
    :param density: density (kg/m^3)
    :type density: array
    :param thickness: thickness of each layer (m). By default is None meaning a semi-infinite medium.
    :type thickness: array
    :param shape_parameterization: method to determine the values of B and g as a function of the wavelength.
        "robledano23", the default, use B=n^2 and a constant defaut g. "constant" uses B0 and g0 parameters for all
        wavelengths. "linear" use the linear relationships as a function of the refractive index deduced from Kokhanosvky
        2004 tables (p61).
    :param g0: asymmetry parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default
        value is 0.86. g0 can be scalar (constant in the snowpack) or an array like the SSA.
    :type g0: array or scalar
    :param B0: absorption enhancement parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The
        default value is 1.6, taken from Libois et al. 2014. B0 can be scalar (constant in the snowpack) or an array like
        the SSA.
    :type B0: array or scalar
    :param impurities: impurities concentration (kg/kg) in each layer. It is either a constant or an array with size
        equal to the number of layers. The array is 1-d if only one type of impurities is used and 2-d otherwise.
    :type impurities: array or scalar
    :param impurities_type: specify the type of impurities. By defaut it is "soot". Otherwise it should be a class (or
        an instance) that defines the density and the imaginary part of the refractive index like the Soot class (see
        tartes.impurities for possible choices and to add new impurities types). It can also be a list of classes if several
        types of impurities are present in the snowpack. In this case, the impurities parameter must be a 2-d array.
    :type impurities_type: object or list of object
    :param soilalbedo: spectral albedo of the underlying layer (no unit). soilalbedo can be a scalar or an array like
        wavelength.
    :type soilalbedo: scalar or array
    :param dir_frac: fraction of directional flux over the total flux (the default is dir_frac = 0 meaning 100% diffuse
        incident flux)
    :type dir_frac: array
    :param diff_method: the method used to compute the diffuse radiation. Possible options are
        1) "aart eq" which perform direct calculation at the 48.2° equivalent angle according to AART,
        2) "integration" which performs the integration of incident radiation over the incidence angle (precise but slow),
        3) "2 streams" which uses diffuse boundary conditions in the 2 stream approximations (inexact for large absorptions,
         not recommended in general).
        The first one is the default, a trade off between speed and accuracy.
    :type diff_method: array or scalar
    :param sza: incident angle of direct light (degrees, 0 means nadir)
    :type sza: scalar
    :param refrac_index: real and imaginary parts of the refractive index for each wavelength or a string: "p2016" for
        Picard et al. 2016, "w2008" for Warren and Brandt 2008 or "w1995" for Warren, 1984 modified as explained on Warren
        web site.
    :type refrac_index: string or tuple of two arrays

    :returns: a tuple with (spectral albedo, direct part of the spectral albedo (reflected part), diffuse part of the
        spectral albedo) for each wavelength."""

    mudir = np.cos(np.deg2rad(sza))

    albedo = tartes(
        wavelength=wavelength,
        SSA=SSA,
        density=density,
        thickness=thickness,
        shape_parameterization=shape_parameterization,
        g0=g0,
        B0=B0,
        impurities=impurities,
        impurities_type=impurities_type,
        soilalbedo=soilalbedo,
        dir_frac=dir_frac,
        diff_method=diff_method,
        infprop_method=infprop_method,
        mudir=mudir,
        refrac_index=refrac_index,
        return_dir_diff=return_dir_diff,
    )

    if albedo.shape == (1,):
        return float(albedo)
    else:
        return albedo


def absorption_profile(
    wavelength: FlexibleArray,
    SSA: FlexibleArray,
    density: Optional[FlexibleArray] = None,
    thickness: Optional[FlexibleArray] = None,
    shape_parameterization: str = 'robledano23',
    g0: FlexibleArray = default_g0,
    B0: Union[str, FlexibleArray] = default_B0,
    impurities: FlexibleArray = 0.0,
    impurities_type: Type[Impurities] = SootSNICAR3,
    refrac_index: Union[str, FlexibleArray, Callable] = 'p2016',
    soilalbedo: FlexibleArray = 0.0,
    dir_frac: FlexibleArray = 0.0,
    totflux: FlexibleArray = 1.0,
    diff_method: str = 'aart eq',
    infprop_method: str = 'delta_eddington',
    sza: FlexibleArray = 0.0,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """compute the energy absorbed in every layer and in the soil. The parameters are the same as for the albedo function. If
    both the albedo and the absorption_profile is needed, a direct call to the tartes function is recommended.

    :param wavelength: wavelength (m)
    :type wavelength: array
    :param SSA: surface specific area (m^2/kg) for each layer
    :type SSA: array
    :param density: density (kg/m^3)
    :type density: array
    :param thickness: thickness of each layer (m). By default is None meaning a semi-infinite medium.
    :type thickness: array
    :param shape_parameterization: method to determine the values of B and g as a function of the wavelength.
        "robledano23", the default, use B=n^2 and a constant defaut g. "constant" uses B0 and g0 parameters for all
        wavelengths. "linear" use the linear relationships as a function of the refractive index deduced from Kokhanosvky
        2004 tables (p61).
    :param g0: asymmetry parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default value is
        0.86. g0 can be scalar (constant in the snowpack) or an array like the SSA.
    :type g0: array or scalar
    :param B0: absorption enhancement parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The
        default value is 1.6, taken from Libois et al. 2014. B0 can be scalar (constant in the snowpack) or an array like the
        SSA.
    :type B0: array or scalar
    :param impurities: impurities concentration (kg/kg) in each layer. It is either a constant or an array with size equal
        to the number of layers. The array is 1-d if only one type of impurities is used and 2-d otherwise.
    :type impurities: array or scalar
    :param impurities_type: specify the type of impurity. By defaut it is "soot". Otherwise it should be a class (or an
        instance) defining the density and the imaginary part of the refractive index like the Soot class (see tartes.impurities
        for possible choices and to add new impurities types). It can also be a list of classes if several impurities type are
        present in the snowpack. In this case, the impurities parameter must be a 2-d array.
    :type impurities_type: object or list of object
    :param soilalbedo: spectral albedo of the underlying (no unit). albedo can be a scalar or an array like wavelength.
    :type soilalbedo: scalar or array
    :param dir_frac: fraction of directional flux over the total flux (the default is dir_frac = 0 meaning 100% diffuse
        incident flux)
    :type dir_frac: array
    :param totflux: total spectral incident flux across the horizontal surface (direct+diffuse). 
        The direct flux can be calculated from the direct beam radiance F0 incoming at the incidence theta_inc with F0 x
        cos(theta_inc). The unit of totflux is at the user's discretion as TARTES internal calculation is relative to
        the incoming flux. Only the returned result is multiplied by totflux. A usual unit is W /m^2 /nm. 
    :type totflux: array
    :param diff_method: the method used to compute the diffuse radiation. Possible options are
        1) "aart eq" which perform direct calculation at the 48.2° equivalent angle according to AART,
        2) "integration" which performs the integration of incident radiation over the incidence angle (precise but slow),
        The first one is the default, a trade off between speed and accuracy.
    :type diff_method: array or scalar
    :param sza: incident angle of direct light (degrees, 0 means nadir)
    :type sza: scalar
    :param refrac_index: real and imaginary parts of the refractive index for each wavelength or a string "p2016" for Picard
        et al. 2016 or w2008 for Warren and Brandt 2008 or w1995 for Warren, 1984 modified as noted on Warren web site.:type
        refrac_index: string or tuple of two arrays

    :returns: spectral absorption in every layer and in the soil. The unit is determined by the
        unit of totflux. The return type is an array with the first dimension for the wavelength and the second for the
        layers + an extra value for the soil. If the wavelength is a scalar, the first
        dimension is squeezed.
    """

    mudir = np.cos(np.deg2rad(sza))

    _, absorption = tartes(
        wavelength=wavelength,
        SSA=SSA,
        density=density,
        thickness=thickness,
        shape_parameterization=shape_parameterization,
        g0=g0,
        B0=B0,
        impurities=impurities,
        impurities_type=impurities_type,
        soilalbedo=soilalbedo,
        dir_frac=dir_frac,
        diff_method=diff_method,
        infprop_method=infprop_method,
        totflux=totflux,
        mudir=mudir,
        refrac_index=refrac_index,
        compute_absorption=True,
    )

    return np.insert(np.cumsum(np.array(thickness)), 0, 0), absorption


def irradiance_profiles(
    wavelength: FlexibleArray,
    z: FlexibleArray,
    SSA: FlexibleArray,
    density: Optional[FlexibleArray] = None,
    thickness: Optional[FlexibleArray] = None,
    shape_parameterization: str = 'robledano23',
    g0: FlexibleArray = default_g0,
    B0: Union[str, FlexibleArray] = default_B0,
    impurities: FlexibleArray = 0.0,
    impurities_type: Type[Impurities] = SootSNICAR3,
    refrac_index: Union[str, FlexibleArray, Callable] = 'p2016',
    soilalbedo: FlexibleArray = 0.0,
    dir_frac: FlexibleArray = 0.0,
    totflux: FlexibleArray = 1.0,
    diff_method: str = 'aart eq',
    infprop_method: str = 'delta_eddington',
    sza: FlexibleArray = 0.0,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """compute the upwelling and downwelling irradiance at every depth z. The parameters are the same as for the
    absorption_profile function plus the depths z. If both the albedo and the absorption_profile is needed, a direct
    call to the tartes function is recommended.

    :param wavelength: wavelength (m)
    :type wavelength: array
    :param z: depths at which the irradiance are calculated (m)
    :type z: array
    :param SSA: surface specific area (m^2/kg) for each layer
    :type SSA: array
    :param density: density (kg/m^3)
    :type density: array
    :param thickness: thickness of each layer (m). By default is None meaning a semi-infinite medium.
    :type thickness: array
    :param shape_parameterization: method to determine the values of B and g as a function of the wavelength.
        "robledano23", the default, use B=n^2 and a constant defaut g. "constant" uses B0 and g0 parameters for all
        wavelengths. "linear" use the linear relationships as a function of the refractive index deduced from Kokhanosvky
        2004 tables (p61).
    :param g0: asymmetry parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default
        value is 0.86. g0 can be scalar (constant in the snowpack) or an array like the SSA.
    :type g0: array or scalar
    :param B0: absorption enhancement parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The
        default value is 1.6, taken from Libois et al. 2014. B0 can be scalar (constant in the snowpack) or an array like
        the SSA.
    :type B0: array or scalar
    :param impurities: impurities concentration (kg/kg) in each layer. It is either a constant or an array with size
        equal to the number of layers. The array is 1-d if only one type of impurities is used and 2-d otherwise.
    :type impurities: array or scalar
    :param impurities_type: specify the type of impurity. By defaut it is "soot". Otherwise it should be a class (or an
        instance) defining the density and the imaginary part of the refractive index like the Soot class (see
        tartes.impurities for possible choices and to add new impurities types). It can also be a list of classes if several
        impurities type are present in the snowpack. In this case, the impurities parameter must be a 2-d array.
    :type impurities_type: object or list of object
    :param soilalbedo: spectral albedo of the underlying (no unit). soilalbedo can be a scalar or an array like
        wavelength.
    :type soilalbedo: scalar or array
    :param dir_frac: fraction of directional flux over the total flux (the default is dir_frac = 0 meaning 100% diffuse
        incident flux) at every wavelength
    :type dir_frac: array
    :param totflux: total spectral incident flux across the horizontal surface (direct+diffuse). 
        The direct flux can be calculated from the direct beam radiance F0 incoming at the incidence theta_inc with F0 x
        cos(theta_inc). The unit of totflux is at the user's discretion as TARTES internal calculation is relative to
        the incoming flux. Only the returned result is multiplied by totflux. A usual unit is W /m^2 /nm. 
    :type totflux: array
    :param diff_method: the method used to compute the diffuse radiation. Possible options are
        1) "aart eq" which perform direct calculation at the 48.2° equivalent angle according to AART,
        2) "integration" which performs the integration of incident radiation over the incidence angle (precise but slow),
        The first one is the default, a trade off between speed and accuracy.
    :type diff_method: array or scalar
    :param sza: solar zenith angle of direct light (degree, 0 means nadir)
    :type sza: scalar
    :param refrac_index: real and imaginary parts of the refractive index for each wavelength or a string "p2016" for
        Picard et al. 2016 or w2008 for Warren and Brandt 2008 or w1995 for Warren, 1984 modified as noted on Warren web
        site.
    :type refrac_index: string or tuple of two arrays

    :returns: a tuple with downwelling and upwelling irradiance profiles. The unit is determined by the
        unit of totflux. The return type is an array with the first dimension for the wavelength and the second for the layers. If
        the wavelength argument is a scalar, the first dimension is squeezed.
    """

    mudir = np.cos(np.deg2rad(sza))

    albedo, down, up = tartes(
        wavelength=wavelength,
        SSA=SSA,
        density=density,
        thickness=thickness,
        shape_parameterization=shape_parameterization,
        g0=g0,
        B0=B0,
        impurities=impurities,
        impurities_type=impurities_type,
        soilalbedo=soilalbedo,
        dir_frac=dir_frac,
        diff_method=diff_method,
        infprop_method=infprop_method,
        totflux=totflux,
        mudir=mudir,
        refrac_index=refrac_index,
        compute_irradiance_profiles=True,
        z=z,
    )

    return down.squeeze(), up.squeeze()


def actinic_profile(
    wavelength: FlexibleArray,
    z: FlexibleArray,
    SSA: FlexibleArray,
    density: Optional[FlexibleArray] = None,
    thickness: Optional[FlexibleArray] = None,
    shape_parameterization: str = 'robledano23',
    g0: FlexibleArray = default_g0,
    B0: Union[str, FlexibleArray] = default_B0,
    impurities: FlexibleArray = 0.0,
    impurities_type: Type[Impurities] = SootSNICAR3,
    refrac_index: Union[str, FlexibleArray, Callable] = 'p2016',
    soilalbedo: FlexibleArray = 0.0,
    dir_frac: FlexibleArray = 0.0,
    totflux: FlexibleArray = 1.0,
    diff_method: str = 'aart eq',
    infprop_method: str = 'delta_eddington',
    sza: FlexibleArray = 0.0,
) -> npt.NDArray[np.floating]:
    """compute the actinic flux at every depth z. The parameters are the same as for the irradiance profile.

    Note that the incoming totflux is the spectral flux across an horizontal surface (a.k.a irradiance). To convert a
    radiance of a direct beam to totflux, the radiance must be multiplied by cos(theta_inc).
    The returned values of this function has the same unit as totflux. 

    If totflux is given in W/m^2/nm, to obtain the result in photons / m^2 / s, the user must divide the returned values
    by the energy of a single photon: H_PLANCK * C_SPEED / wavelength(nm).

    :param wavelength: wavelength (m)
    :type wavelength: array
    :param z: depths at which the irradiance are calculated (m)
    :type z: array
    :param SSA: surface specific area (m^2/kg) for each layer
    :type SSA: array
    :param density: density (kg/m^3)
    :type density: array
    :param thickness: thickness of each layer (m). By default is None meaning a semi-infinite medium.
    :type thickness: array
    :param shape_parameterization: method to determine the values of B and g as a function of the wavelength.
        "robledano23", the default, use B=n^2 and a constant defaut g. "constant" uses B0 and g0 parameters for all
        wavelengths. "linear" use the linear relationships as a function of the refractive index deduced from Kokhanosvky
        2004 tables (p61).
    :param g0: asymmetry parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default value is
        0.86. g0 can be scalar (constant in the snowpack) or an array like the SSA.
    :type g0: array or scalar
    :param B0: absorption enhancement parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The
        default value is 1.6, taken from Libois et al. 2014. B0 can be scalar (constant in the snowpack) or an array like the
        SSA.
    :type B0: array or scalar
    :param impurities: impurities concentration (kg/kg) in each layer. It is either a constant or an array with size equal
        to the number of layers. The array is 1-d if only one type of impurities is used and 2-d otherwise.
    :type impurities: array or scalar
    :param impurities_type: specify the type of impurity. By defaut it is "soot". Otherwise it should be a class (or an
        instance) defining the density and the imaginary part of the refractive index like the Soot class (see tartes.impurities
        for possible choices and to add new impurities types). It can also be a list of classes if several impurities type are
        present in the snowpack. In this case, the impurities parameter must be a 2-d array.
    :type impurities_type: object or list of object
    :param soilalbedo: spectral albedo of the underlying (no unit). soilalbedo can be a scalar or an array like wavelength.
    :type soilalbedo: scalar or array
    :param dir_frac: fraction of directional flux over the total flux (the default is dir_frac = 0 meaning 100% diffuse
        incident flux) at every wavelength
    :type dir_frac: array
    :param diff_method: the method used to compute the diffuse radiation. Possible options are
        1) "aart eq" which perform direct calculation at the 48.2° equivalent angle according to AART,
        2) "integration" which performs the integration of incident radiation over the incidence angle (precise but slow),
        The first one is the default, a trade off between speed and accuracy.
    :type diff_method: array or scalar
    :param totflux: total spectral incident flux across the horizontal surface (direct+diffuse). 
        The direct flux can be calculated from the direct beam radiance F0 incoming at the incidence theta_inc with F0 x
        cos(theta_inc). The unit of totflux is at the user's discretion as TARTES internal calculation is relative to
        the incoming flux. Only the returned result is multiplied by totflux. A usual unit is W /m^2 /nm.
    :type totflux: array
    :param sza: solar zenith angle of direct light (degree, 0 means nadir)
    :type sza: scalar
    :param refrac_index: real and imaginary parts of the refractive index for each wavelength or a string "p2016" for Picard
        et al. 2016 or w2008 for Warren and Brandt 2008 or w1995 for Warren, 1984 modified as noted on Warren web site.
    :type refrac_index: string or tuple of two arrays

    :returns: the actinic flux profile. The unit is determined by the
        unit of totflux. The return type is an array with the first dimension for the wavelength and the
        second for the layers. If the wavelength argument is a scalar, the first dimension is squeezed.
    """

    mudir = np.cos(np.deg2rad(sza))

    albedo, actinic = tartes(
        wavelength=wavelength,
        SSA=SSA,
        density=density,
        thickness=thickness,
        shape_parameterization=shape_parameterization,
        g0=g0,
        B0=B0,
        impurities=impurities,
        impurities_type=impurities_type,
        soilalbedo=soilalbedo,
        dir_frac=dir_frac,
        diff_method=diff_method,
        infprop_method=infprop_method,
        totflux=totflux,
        mudir=mudir,
        refrac_index=refrac_index,
        compute_actinic_profile=True,
        z=z,
    )

    return actinic.squeeze()


def atmospheric_incident_spectrum(
    wavelength: FlexibleArray, sza: float, cloud_optical_depth: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """return incident total flux and dir_frac computed with SBDART and ready to use for TARTES.
    offers much less control than directly calling SBDART or other atmospheric radative transfer model.

    .. note::
         This function requires atmosrt package to be installed (https://github.com/ghislainp/atmosrt).

    :param wavelength: wavelength (m)
    :type wavelength: array
    :param sza: solar zenith angle of direct light (degree, 0 means nadir)
    :type sza: scalar
    :param cloud_optical_depth: Optical depth of the cloud layer.
    :type cloud_optical_depth: scalar

    :returns: a tuple with the totflux and the dir_frac ratio"""

    import atmosrt  # here for lazy import.

    settings = dict(atmosrt.settings.defaults)

    wavelength = np.atleast_1d(wavelength)
    settings['lower_limit'] = np.min(wavelength) * 1e6
    settings['upper_limit'] = np.max(wavelength) * 1e6
    settings['resolution'] = max(np.min(np.diff(wavelength)) * 1e6, 0.001)  # um

    settings['cloud_optical_depth'] = cloud_optical_depth
    model = atmosrt.SBdart(
        settings,
        time=datetime.datetime(2020, 3, 20, 12, 0),
        latitude=sza,
        longitude=0.0000,
    )

    spec = model.spectrum()

    totflux = np.interp(wavelength, spec.index * 1e-6, spec['global'])
    direct = np.interp(wavelength, spec.index * 1e-6, spec['direct'])
    return totflux, direct / totflux


#######################################################################################################
#
# The following functions are the core of TARTES but are as convenient to use as the previous one.
# Use it when the previous functions are insufficient or in case of performance issue. Experts only!
#
#
#######################################################################################################


def shape_parameter_variations(
    shape_parameterization: str,
    nr: ArrayOrScalar,
    g0: ArrayOrScalar,
    y0: ArrayOrScalar,
    W0: ArrayOrScalar,
    B0: ArrayOrScalar,
) -> tuple[ArrayOrScalar, ArrayOrScalar, ArrayOrScalar, ArrayOrScalar, ArrayOrScalar]:
    """compute shape parameter variations as a function of the the refraction index with respect to the value in the
    visible range. These variation equations were obtained for sphere (Light Scattering Media Optics, Kokhanovsky, A.,
    p.61) but should also apply to other shapes in a first approximation.
    see doc Section 2

    :param nr: refractive index (no unit). It is a constant array recalculated for each spectral resolution
    :type nr: array
    :param g0: asymmetry parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit)
    :type g0: scalar
    :param B0: absorption enhancement parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit)
    :type B0: scalar
    :param W0: Value of W of snow grains at nr=1.3 (no unit)
    :type W0: scalar
    :param y0: Value of y of snow grains at nr=1.3 (no unit)
    :type y0: scalar

    :returns: spectral parameters necessary to compute the asymmetry parameter and single scattering albedo of snow. For
        now, those parameters do not evolve with time. They depend on shape only

    """

    if shape_parameterization == 'robledano23':
        B = nr**2
        ginf = g00 = robledano23_g0
    elif shape_parameterization == 'constant':
        B = B0
        ginf = g00 = g0
    elif shape_parameterization == 'linear':
        ginf = 0.9751 - 0.105 * (nr - 1.3)
        g00 = g0 - 0.38 * (nr - 1.3)
        B = B0 + 0.4 * (nr - 1.3)
    else:
        raise Exception(
            'The argument shape_parameterization is incorrect. Refer to the documentation for possible options'
        )

    W = W0 + 0.17 * (nr - 1.3)
    y = y0 + 0.752 * (nr - 1.3)

    return ginf, g00, B, W, y


def impurities_co_single_scattering_albedo(
    wavelength: ArrayOrScalar,
    SSA: ArrayOrScalar,
    impurities_content: ArrayOrScalar,
    impurities_type,
):
    """return the spectral absorption of one layer due to the impurities
    see doc Section 2.6

    :param wavelength: all wavelengths of the incident light (m)
    :type wavelength: array
    :param SSA: snow specific surface area (m^2/kg)
    :param impurities: impurities is a dictionnary where keys are impurity type ("soot" or "hulis") and values are a
        2-element array containing density (kg/m^3) and content (g/g)
    :type impurities: dict

    :returns: co single scattering albedo of impurities"""

    if impurities_content is None:
        return 0

    def one_species_co_albedo(
        wavelength: ArrayOrScalar,
        SSA: ArrayOrScalar,
        impurities_content: ArrayOrScalar,
        impurities_type,
    ):
        """return the co-albedo for on species"""
        if impurities_content <= 0:
            return 0.0

        mae_impurities = impurities_type.MAE(wavelength)  # in m^-1 / (kg m^-3)

        # density could be remove because it is
        # return 2/(density*SSA) * mae_impurities * impurities_content * density
        return 2.0 / SSA * mae_impurities * impurities_content

    if hasattr(impurities_type, '__iter__'):
        assert isinstance(impurities_content, Sequence)
        cossalb = 0.0
        for i, species in enumerate(impurities_type):
            cossalb += one_species_co_albedo(wavelength, SSA, impurities_content[i], species)
    else:
        cossalb = one_species_co_albedo(wavelength, SSA, impurities_content, impurities_type)

    return cossalb


def single_scattering_optical_parameters(
    wavelength: ArrayOrScalar,
    refrac_index: Optional[Union[str, ArrayOrScalar, Callable]],
    SSA: ArrayOrScalar,
    impurities_content: Optional[ArrayOrScalar] = None,
    impurities_type: Optional[Type[Impurities]] = None,
    shape_parameterization: str = 'robledano23',
    g0: ArrayOrScalar = default_g0,
    B0: Union[str, ArrayOrScalar] = default_B0,
) -> tuple[ArrayOrScalar, ArrayOrScalar]:
    """return single scattering parameters of one layer
    see doc Section 2.3, 2.5, 2.6

    :param wavelength: wavelength (m)
    :type wavelength: array
    :param refrac_index: real and imag part of the refractive index (no unit)
    :type refrac_index: array
    :param SSA: snow specific surface area (m^2/kg) of one layer
    :type SSA: scalar
    :param impurities: impurities is a dictionnary where keys are impurity type ("soot" or "hulis") and values are a
        2-element array containing density (kg/m^3) and content (g/g)
    :type impurities: dict

    :returns: total single scattering albedo and asymmetry factor"""
    if refrac_index == 'w2008':
        # should be cached when the same wavelengths are used
        refrac_index = refice2008(wavelength)
    elif refrac_index == 'w1995':
        # should be cached when the same wavelengths are used
        refrac_index = refice1995(wavelength)
    elif (refrac_index is None) or (refrac_index == 'p2016'):
        # should be cached when the same wavelengths are used
        refrac_index = refice2016(wavelength)
    elif callable(refrac_index):
        # should be cached when the same wavelengths are used
        refrac_index = refrac_index(wavelength)
    else:
        # assume refrac_index is a number/array
        pass
    assert isinstance(refrac_index, Sequence)

    nr, ni = refrac_index  # determination of ice refractive index
    c = 24.0 * np.pi * ni / (917.0 * wavelength) / SSA  # 917 is ice density

    ginf, g00, B, W, y = shape_parameter_variations(shape_parameterization, nr, g0, default_y0, default_W0, B0)

    # calculation of the spectral asymmetry parameter of snow
    g = ginf - (ginf - g00) * np.exp(-y * c)

    # co- single scattering albedo of pure snow
    phi = 2.0 / 3 * B / (1 - W)
    cossalb = 0.5 * (1 - W) * (1 - np.exp(-c * phi))

    # adding co- single scattering albedo for impureties
    cossalb += impurities_co_single_scattering_albedo(wavelength, SSA, impurities_content, impurities_type)

    ssalb = 1.0 - cossalb

    return ssalb, g


def infinite_medium_optical_parameters_delta_eddington(
    ssalb: ArrayOrScalar, g: ArrayOrScalar
) -> tuple[ArrayOrScalar, ArrayOrScalar]:
    """return albedo and ke using Delta-Eddington Approximation (The Delta-Eddington Approximation of
    Radiative Flux Transfer, Jospeh et al (1976)).
    The fluxes in the snowpack depend on these 2 quantities
    see doc section 1.4

    :param ssalb: single scattering albedo (no unit)
    :type ssalb: array
    :param g: asymmetry factor (no unit)
    :type g: array

    :returns: albedo and normalised AFEC
    """

    ssalb_star = ssalb * (1 - g**2) / (1 - g**2 * ssalb)
    g_star = g / (1 + g)

    # Jimenez-Aquino, J. and Varela, J. R., (2005)
    gamma1 = 0.25 * (7 - ssalb_star * (4 + 3 * g_star))
    gamma2 = np.maximum(-0.25 * (1 - ssalb_star * (4 - 3 * g_star)), 0.0001)

    ke = np.sqrt(gamma1**2 - gamma2**2)

    # albedo = (gamma1 - ke) / gamma2
    albedo = gamma2 / (gamma1 + ke)

    return albedo, ke


def infinite_medium_optical_parameters_aart(
    ssalb: ArrayOrScalar, g: ArrayOrScalar
) -> tuple[ArrayOrScalar, ArrayOrScalar]:
    """return albedo and ke for a semi-infinite medium using AART equation (Kokhanovsky and Zege, 2004).
    The equations are also derived in Libois et al. 2013, Eq 8 and Eq 9. Only (1-g) is replaced by
    (1-ssalb * g) which is the case of snow makes very little difference, but is more conform to the original
    formulation.

    :param ssalb: single scattering albedo (no unit)
    :type ssalb: array
    :param g: asymmetry factor (no unit)
    :type g: array

    :returns: albedo and normalised AFEC
    """

    g_star = g  # / (1 + g)
    ssalb_star = ssalb  # * (1 - g**2) / (1 - g**2 * ssalb)

    ke = np.sqrt(3 * (1.0 - ssalb_star) * (1.0 - ssalb_star * g_star))
    albedo = np.exp(-4 * np.sqrt((1.0 - ssalb_star) / (3 * (1.0 - g_star))))

    return albedo, ke


# register the possible option for the infinite_medium_optical_parameters calculation
infinite_medium_optical_parameters = {
    'delta_eddington': infinite_medium_optical_parameters_delta_eddington,
    'aart': infinite_medium_optical_parameters_aart,
}


def taustar_vector_delta_eddington(
    sigext: ArrayOrScalar,
    thickness: npt.NDArray,
    ssalb: ArrayOrScalar,
    g: ArrayOrScalar,
    ke: ArrayOrScalar,
):
    """compute the taustar and dtaustar of the snowpack for the delta-eddington formulation,
    the optical depth of each layer and cumulated optical depth
    see doc Section 1.2, 1.8, 2.4

    :param sigext: extinction coefficient
    :type sigext: array
    :param thickness: thickness of each layers (m)
    :type thickness: array
    :param ssalb: single scattering albedo (no unit)
    :type ssalb: array
    :param g: asymmetry factor (no unit)
    :type g: array
    :param ke: delta Eddington asymptotic flux extinction coefficient (no unit)
    :type ke: array

    :returns: optical depth of each layer (unbounded + bounded) and cumulated optical depth (no unit)
    """
    # delta-Eddington variable change
    dtaustar_ub = sigext * thickness[np.newaxis, :] * (1 - ssalb * g**2)

    maximum_optical_depth_per_layer = 200.0
    dtaustar = np.minimum(dtaustar_ub, maximum_optical_depth_per_layer / ke)
    # this is a dirty hack and causes problem with the irradiance profile calculation.
    # This is reason why we need to return the unbunded and the bunded dtaustar.
    # In practice, it is safe (but dirty) as a layer with optical >200 has a null transmittance

    taustar = np.cumsum(dtaustar, axis=1)

    return dtaustar_ub, dtaustar, taustar


def taustar_vector_aart(
    sigext: ArrayOrScalar,
    thickness: npt.NDArray,
    ssalb: ArrayOrScalar,
    g: ArrayOrScalar,
    ke: ArrayOrScalar,
):
    """compute the taustar and dtaustar of the snowpack for the AART formulation, the optical depth of each layer and
    cumulated optical depth.
    see doc Section 1.2, 1.8, 2.4

    :param sigext: extinction coefficient
    :type sigext: array
    :param thickness: thickness of each layers (m)
    :type thickness: array
    :param ssalb: single scattering albedo (no unit)
    :type ssalb: array
    :param g: asymmetry factor (no unit)
    :type g: array
    :param ke: delta Eddington asymptotic flux extinction coefficient (no unit)
    :type ke: array

    :returns: optical depth of each layer (unbounded + bounded) and cumulated optical depth (no unit)
    """
    # delta-Eddington variable change
    dtaustar_ub = sigext * thickness[np.newaxis, :]

    maximum_optical_depth_per_layer = 200.0
    dtaustar = np.minimum(dtaustar_ub, maximum_optical_depth_per_layer / ke)
    # this is a dirty hack and causes problem with the irradiance profile calculation.
    # This is reason why we need to return the unbunded and the bunded dtaustar.
    # In practice, it is safe (but dirty) as a layer with optical >200 has a null transmittance

    taustar = np.cumsum(dtaustar, axis=1)

    return dtaustar_ub, dtaustar, taustar


# register the possible option for the infinite_medium_optical_parameters calculation
taustar_vector = {
    'delta_eddington': taustar_vector_delta_eddington,
    'aart': taustar_vector_aart,
}


class Streams(object):
    mu: npt.NDArray
    nstreams_dir: int
    nstreams_diff: int
    compute_2stream_diff: bool
    return_dir_diff: bool
    dir_frac: ArrayOrScalar

    def __init__(
        self,
        mudir: ArrayOrScalar,
        dir_frac: ArrayOrScalar,
        diff_method: str,
        return_dir_diff: bool,
    ):
        dir_frac = np.array(dir_frac)
        mu_list = list(np.atleast_1d(mudir)) if np.any(dir_frac > 0) else []

        self.return_dir_diff = return_dir_diff
        self.nstreams_dir = len(mu_list)
        self.nstreams_diff = 0
        self.compute_2stream_diff = False

        if np.any(dir_frac < 1):  # we need diffuse
            if diff_method == 'aart eq':
                # solution of 3/7 (1+2 \cos(\theta)) = 1
                mu_list.append((7 / 3 - 1) / 2)  # about 48.2°
                self.nstreams_diff = 1
                self.compute_2stream_diff = False
            elif diff_method in 'integration':
                nstreams_integration = 128
                mu_list += list(np.arange(1, 0, -1 / nstreams_integration))
                self.nstreams_diff = nstreams_integration
                self.compute_2stream_diff = False
            elif diff_method == '2 streams':
                self.nstreams_diff = 1
                self.compute_2stream_diff = True
            else:
                raise Exception(f"the method to compute diffuse '{diff_method}' radiation is not recognized.")

        self.mu = np.array(mu_list)

    @property
    def nstreams_return(self):
        return max(1, self.nstreams_dir + 1 if self.return_dir_diff else self.nstreams_dir)

    def process_output(self, x: npt.NDArray, mu: npt.NDArray, dir_frac: float) -> npt.NDArray[np.floating]:

        if self.nstreams_diff > 1:
            # integration
            assert not self.compute_2stream_diff  # both are incompatible

            x[..., self.nstreams_dir] = np.mean(
                x[..., self.nstreams_dir :] * mu[np.newaxis, self.nstreams_dir :],
                axis=-1,
            ) / np.mean(mu[self.nstreams_dir :])
            x = x[..., : self.nstreams_dir + 1]  # remove the remaining part

        # x ..., diff is the last item in the x ..., array
        if not self.return_dir_diff and (self.nstreams_diff > 0):
            # weight the direct x(..., s) and the diffuse x
            if self.nstreams_dir > 0:
                x = dir_frac * x[..., :-1] + (1 - dir_frac) * x[..., -1:]
            else:
                x = x  # do nothing
        return x


def two_stream_matrix(
    layeralbedo: npt.NDArray[np.floating],
    soilalbedo: float,
    ke: npt.NDArray[np.floating],
    dtaustar: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """compute the matrix describing the boundary conditions at one wavelength.
    see doc Section 1.5

    :param layeralbedo: infinite albedo of each layers (no unit)
    :type layeralbedo: array
    :param soilalbedo: albedo of the bottom layer (no unit)
    :type soilalbedo: scalar
    :param ke: delta-eddington AFEC (no unit)
    :type ke: array
    :param dtaustar: optical depth (no unit)
    :type dtaustar: array

    :returns: tri-diagonal of the boundary matrix"""

    nlyr = len(dtaustar)

    f_diag = np.exp(-ke * dtaustar)

    Dm = np.zeros(2 * nlyr)
    Dm[0:-2:2] = (1 - layeralbedo[0:-1] * layeralbedo[1:]) * f_diag[0:-1]
    Dm[1:-1:2] = (1 / layeralbedo[0:-1] - layeralbedo[0:-1]) / f_diag[0:-1]

    D = np.zeros(2 * nlyr)
    D[1:-2:2] = (1 - layeralbedo[1:] / layeralbedo[0:-1]) / f_diag[0:-1]
    D[2:-1:2] = layeralbedo[0:-1] - layeralbedo[1:]

    Dp = np.zeros(2 * nlyr)
    Dp[2:-1:2] = layeralbedo[1:] * layeralbedo[1:] - 1
    Dp[3::2] = layeralbedo[0:-1] - 1.0 / layeralbedo[1:]

    # Bottom and top layer
    Dp[1] = 1
    D[0] = 1
    Dm[-2] = (layeralbedo[-1] - soilalbedo) * f_diag[-1]
    D[-1] = (1.0 / layeralbedo[-1] - soilalbedo) / f_diag[-1]

    d = np.array([Dp, D, Dm])

    return d


def Gp_Gm_vectors_delta_eddington(
    ssalb: npt.NDArray, ke: npt.NDArray, g: npt.NDArray, mu: npt.NDArray
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """return Gp and Gm vectors at one wavelength. 

    Note that F0 * mu0 (ie totflux) is factorized in Tartes code, it does
    not appear in G terms, neither in vector V.

    :param ssalb: single scattering albedo of each layer (no unit)
    :type ssalb: array
    :param ke: delta-eddington AFEC (no unit)
    :type ke: array
    :param g: asymmetry factor  of each layer (no unit)
    :type g: array
    :param mu: cosine of the incident angle

    :returns: Gp and Gm"""
    mu = mu[None, :]
    ssalb = ssalb[:, None]
    g = g[:, None]
    ke = ke[:, None]

    g_star = g / (1 + g)
    ssalb_star = ssalb * (1 - g**2) / (1 - g**2 * ssalb)

    gamma1 = 0.25 * (7 - ssalb_star * (4 + 3 * g_star))
    gamma2 = np.maximum(-0.25 * (1 - ssalb_star * (4 - 3 * g_star)), 0.0001)

    gamma3 = 0.25 * (2 - 3 * g_star * mu)
    gamma4 = 0.25 * (2 + 3 * g_star * mu)

    # G = mu**2 * ssalb_star / ((ke * mu)**2 - 1)
    G = mu * ssalb_star / ((ke * mu) ** 2 - 1)  # mu instead of mu**2 because we have factorized out mu*Fdot
    Gp = G * ((gamma1 - 1 / mu) * gamma3 + gamma2 * gamma4)
    Gm = G * ((gamma1 + 1 / mu) * gamma4 + gamma2 * gamma3)

    # AART verison of Gp, Gm

    # En effet les G+ et G- vont poser problème.
    # Il faudrait déjà regarder en situation d'éclairement vraiment diffus (Fdir = 0), quand il n'y a pas de G.
    # Aussitôt qu'il y a des G, Il faudrait les modifier.
    # Il faut exprimer l'albédo semi-infini du two-stream -> alb = (alb_diff A + G+)/(A+G-+Fdir), et l'égaler à l'albédo direct de ART.
    # Sachant que A+G- = 0
    # Il faudrait avoir une relation constitutive en plus entre G+ et G-. Sachant que gamma3+gamma4=1 (conservation
    # d'énergie), et que l'on peut prendre les relations suivantes du Eddington:
    # gamma1+gamma2 = 3/2(1-g) et gamma4-gamma3 = 3/2g mu0
    # Imposons par exemple aux nouveaux G- et G+ d'être tels que leur somme vaut celle d'Eddington
    # Quand on somme G-+G+ d'Eddington il reste X(gamma1+gamma2+1/mu0(gamma4-gamma3) = 3X/2, aux erreurs de calcul près.
    # Où X est le gros préfacteur commun à G+ et G-
    # Bref si tu veux essayer qqch tu pourrais choisir G+ tel que

    # G+ = alb_dir*mu0*F0/alb_diff+3X/2
    # G- = -alb_dir*mu0*F0/alb_diff

    return Gp, Gm


def Gp_Gm_vectors_aart(
    ssalb: npt.NDArray, ke: npt.NDArray, g: npt.NDArray, mu: npt.NDArray
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """return Gp and Gm vectors at one wavelength

    Note that F0 * mu0 (ie totflux) is factorized in Tartes code, it does
    not appear in G terms, neither in vector V.

    :param ssalb: single scattering albedo of each layer (no unit)
    :type ssalb: array
    :param ke: delta-eddington AFEC (no unit)
    :type ke: array
    :param g: asymmetry factor  of each layer (no unit)
    :type g: array
    :param mu: cosine of the incident angle

    :returns: Gp and Gm"""

    mu = mu[None, :]
    ssalb = ssalb[:, None]
    g = g[:, None]
    ke = ke[:, None]

    # the AART version does not use the delta-scaled quantities
    # g_star = g #/ (1 + g)
    # ssalb_star = ssalb #* (1 - g**2) / (1 - g**2 * ssalb)

    G0 = mu * ssalb / ((ke * mu) ** 2 - 1)  # mu instead of mu**2 because we have factorized out mu*Fdot

    a = 4 * np.sqrt((1 - ssalb) / (3 * (1 - g)))
    albedo_diff = np.exp(-a)
    albedo_dir = np.exp(-a * 3.0 / 7 * (1 + 2 * mu))

    sum_Gm_Gp = 3 / 2 * G0 * (1 - ssalb * g + g)
    Gm = (sum_Gm_Gp - albedo_dir) / (albedo_diff + 1)
    Gp = sum_Gm_Gp - Gm

    return Gp, Gm


# register the possible option for the Gp_Gp_vectors calculation
Gp_Gm_vectors = {
    'delta_eddington': Gp_Gm_vectors_delta_eddington,
    'aart': Gp_Gm_vectors_aart,
}


def two_stream_vector(
    layeralbedo: npt.NDArray,
    soilalbedo: ArrayOrScalar,
    dtaustar: npt.NDArray,
    taustar: npt.NDArray,
    Gm: npt.NDArray,
    Gp: npt.NDArray,
    mu: npt.NDArray,
    streams: Streams,
) -> npt.NDArray[np.floating]:
    """compute the vector V for the boundary conditions
    see doc Section 1.5

    Note that F0 * mu0 (ie totflux) is factorized in Tartes code, it does
    not appear in G terms, neither in vector V.

    :param layeralbedo: albedo of the layer if it was infinite
    :type layeralbedo: array
    :param soilalbedo: albedo of the bottom layer
    :type soilalbedo: scalar
    :param dtaustar: optical depth
    :type dtaustar: array
    :param taustar: optical depth
    :type taustar: array
    :param Gm: coefficients Gm calculated for each layer
    :type Gm: array
    :param Gp: coefficients Gm calculated for each layer
    :type Gp: array
    :param mu: cosine of the incidence angle
    :type mu: array
    :param compute_2stream_diff: compute 2-stream diff or not
    :type compute_2stream_diff: bool

    :returns: vector V"""
    mu = mu[np.newaxis, :]

    nlyr = len(taustar)
    vect = np.empty((2 * nlyr, mu.shape[1]))

    vect[0] = -Gm[0]
    if nlyr > 1:
        dGp = np.diff(Gp, axis=0)
        dGm = np.diff(Gm, axis=0)
        vect[1:-2:2] = (dGm - layeralbedo[1:, np.newaxis] * dGp) * np.exp(-taustar[0:-1, np.newaxis] / mu)
        vect[2:-1:2] = (dGp - layeralbedo[0:-1, np.newaxis] * dGm) * np.exp(-taustar[0:-1, np.newaxis] / mu)
    vect[-1] = (soilalbedo * (Gm[-1] + 1) - Gp[-1]) * np.exp(
        -taustar[-1] / mu
    )  # 1 is due to the factorization of mu*Fodot

    if streams.compute_2stream_diff:
        v = np.zeros((2 * nlyr, 1))
        v[0] = 1
        vect = np.append(vect, v, axis=1)

    return vect


def solve_two_stream(
    dmatrix: npt.NDArray, vect: npt.NDArray, layeralbedo: npt.NDArray
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """solve the two stream linear system for vect.
    see doc Section 1.5

    :param dmatrix: two-stream matrix  M
    :type dmatrix: matrix
    :param vect: two-stream vector V
    :type vect: array or None
    :param layeralbedo: albedo of the layers if it was infinite
    :type layeralbedo: array

    :returns: vector X, unpacked solution vectors for coefficients A and B"""
    # solve the two stream linear system

    solution0 = solve_banded((1, 1), dmatrix, vect)

    assert solution0.shape[0] == 2 * len(layeralbedo), f'problem with the shape of solution0: {solution0.shape}'
    assert len(solution0.shape) in [
        1,
        2,
    ], f'problem with the shape of solution0: {solution0.shape}'

    solution_A = solution0[:-1:2]
    solution_B = solution0[1::2]

    if len(solution0.shape) == 2:
        layeralbedo = layeralbedo[:, np.newaxis]
    solution_C = solution_A * layeralbedo
    solution_D = solution_B / layeralbedo

    # TODO: use an object "Solution"
    return solution_A, solution_B, solution_C, solution_D


def get_albedo(
    solutions: Sequence[npt.NDArray],
    Gp: npt.NDArray,
    mu: npt.NDArray,
    dir_frac: float,
    streams: Streams,
) -> npt.NDArray[np.floating]:
    """compute the albedo of the snowpack at one wavelength

    :param solutions: coefficients A and B for each layer that allow the analytical calculation of the flux in the snowpack
    :type solutions: array
    :param Gp: coefficients Gm calculated for each layer
    :type Gp: array
    :param mu: cosine of the incident angle of intensity
    :type mu: scalar

    :returns: albedo at one wavelength"""

    if solutions is None:
        return 0

    solution_C = solutions[2]
    solution_D = solutions[3]

    albedo = solution_C[0] + solution_D[0]
    assert albedo.shape[0] == streams.nstreams_dir + streams.nstreams_diff

    if streams.compute_2stream_diff:
        albedo[:-1] += Gp[0]
    else:
        albedo += Gp[0]

    return streams.process_output(albedo, mu, dir_frac)


def get_absorption_profile(
    solutions: Sequence[npt.NDArray],
    ke: npt.NDArray,
    dtaustar: npt.NDArray,
    taustar: npt.NDArray,
    Gm: npt.NDArray,
    Gp: npt.NDArray,
    mu: npt.NDArray,
    dir_frac: float,
    streams: Streams,
) -> npt.NDArray[np.floating]:
    """compute energy absorption for each layer at one wavelength

    :param solutions: coefficients A and B for each layer that allow the analytical calculation of the flux in the snowpack
    :type solutions: array
    :param ke: delta-eddington AFEC in each layer (no unit)
    :type ke: array
    :param dtaustar: optical depth of each layer
    :type dtaustar: array
    :param taustar: cumulated optical depth at each interface
    :type taustar: array
    :param Gm: coefficients Gm calculated for each layer
    :type Gm: array
    :param Gp: coefficients Gm calculated for each layer
    :type Gp: array
    :param mu: cosine of the incident angle of intensity
    :type mu: scalar

    :returns: energy absorbed by each layer (same as totux, i.e. usually W/m^2 /nm). To get W / m3 / nm, please divide
        by layer thickness.
    """

    # compute energy absoprtion profile

    if solutions is None:
        return 0

    absprofile = np.zeros((len(taustar), len(mu)))

    solution_A, solution_B, solution_C, solution_D = solutions
    mu = mu[np.newaxis, :]
    taustar = taustar[:, np.newaxis]
    dtaustar = dtaustar[:, np.newaxis]
    ke = ke[:, np.newaxis]

    assert not streams.compute_2stream_diff

    # surface Layer
    absprofile[0] = (1 - (solution_C[0] + solution_D[0] + Gp[0])) + (
        (
            solution_C[0] * np.exp(-ke[0] * dtaustar[0])
            + solution_D[0] * np.exp(ke[0] * dtaustar[0])
            + Gp[0] * np.exp(-taustar[0] / mu)
        )
        - (
            solution_A[0] * np.exp(-ke[0] * dtaustar[0])
            + solution_B[0] * np.exp(ke[0] * dtaustar[0])
            + Gm[0] * np.exp(-taustar[0] / mu)
            + 1 * np.exp(-taustar[0] / mu)
        )
    )

    dexp = np.exp(-taustar[1:] / mu) - np.exp(-taustar[0:-1] / mu)

    expp = np.exp(ke[1:] * dtaustar[1:])
    expm = np.exp(-ke[1:] * dtaustar[1:])
    fdu = solution_C[1:] * (expm - 1) + solution_D[1:] * (expp - 1) + Gp[1:] * dexp
    fdd = solution_A[1:] * (expm - 1) + solution_B[1:] * (expp - 1) + (Gm[1:] + 1) * dexp  # +1 was mu before

    absprofile[1:] = fdu - fdd

    return streams.process_output(absprofile, mu, dir_frac)


def get_soil_absorption(
    solutions: Sequence[npt.NDArray],
    ke: npt.NDArray,
    dtaustar: npt.NDArray,
    taustar: npt.NDArray,
    Gm: npt.NDArray,
    mu: npt.NDArray,
    dir_frac: float,
    streams: Streams,
    soilalbedo: float,
) -> npt.NDArray[np.floating]:
    """compute the energy absorbed by the soil at one wavelength

    :param solutions: coefficients A and B for each layer that allow the analytical calculation of the flux in the snowpack
    :type solutions: array
    :param ke: delta-eddington AFEC in each layer (no unit)
    :type ke: array
    :param dtaustar: optical depth of each layer
    :type dtaustar: array
    :param taustar: cumulated optical depth at each interface
    :type taustar: array
    :param Gm: coefficients Gm calculated for each layer
    :type Gm: array
    :param mu: cosine of the incident angle of intensity
    :type mu: scalar
    :param soilalbedo: soil albedo at that wavelength (no unit)
    :type soilalbedo: scalar

    :returns: energy absorbed by the soil at one wavelength. Same unit as totflux (usually W / m^2 / nm)
    """

    if solutions is None:
        return 0

    solution_A = solutions[0]
    solution_B = solutions[1]

    mu = mu[np.newaxis, :]
    taustar = taustar[:, np.newaxis]
    dtaustar = dtaustar[:, np.newaxis]
    ke = ke[:, np.newaxis]

    direct_only = 0 if streams.compute_2stream_diff else 1

    # Soil absorption
    soil_abs = (1 - soilalbedo) * (
        solution_A[-1] * np.exp(-ke[-1] * dtaustar[-1])
        + solution_B[-1] * np.exp(ke[-1] * dtaustar[-1])
        + direct_only * (Gm[-1] + 1) * np.exp(-taustar[-1] / mu)
    )  # + 1 was mu before

    return streams.process_output(soil_abs, mu, dir_frac)


class DepthGenerator(object):
    def __init__(self, z: ArrayOrScalar, thickness: npt.NDArray[np.floating]):
        self.z = np.array(z)
        thickness = np.atleast_1d(thickness)
        self.thickness_total = np.zeros(len(thickness) + 1)
        self.thickness_total[1:] = np.cumsum(thickness)

        # number of the layer (1,2...)
        self.nearest_layer = np.searchsorted(self.thickness_total, z, side='right')

        bottom = z == self.thickness_total[-1]  # if very close to the bottom...
        self.nearest_layer[bottom] = len(self.thickness_total) - 1  # take the last layer
        self.nearest_layer[z > self.thickness_total[-1]] = -1

    def __call__(
        self, dtaustar_ub: npt.NDArray[np.floating], taustar: npt.NDArray[np.floating]
    ) -> Generator[tuple[int, int, np.floating, np.floating], None, None]:
        # it is probably possible to optimize this loop (-> array calculation)
        for nz0, z0 in enumerate(self.z):
            nl = self.nearest_layer[nz0]
            if nl < 0:  # skip, we are below the bottom interface
                continue
            dtaustar_z = (
                (z0 - self.thickness_total[nl - 1])
                / (self.thickness_total[nl] - self.thickness_total[nl - 1])
                * dtaustar_ub[nl - 1]
            )

            taustar_z = dtaustar_z
            if nl > 1:
                taustar_z += taustar[nl - 2]

            yield nz0, nl, dtaustar_z, taustar_z


def get_irradiance_profiles(
    solutions: Sequence[npt.NDArray],
    ke: npt.NDArray,
    dtaustar_ub: ArrayOrScalar,
    taustar: npt.NDArray,
    Gp: npt.NDArray,
    Gm: npt.NDArray,
    mu: npt.NDArray,
    dir_frac: float,
    streams: Streams,
    depth_generator: DepthGenerator,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    # compute the profile of downward intensity

    down_irr_profile = np.zeros((len(depth_generator.z), len(mu)))
    up_irr_profile = np.zeros_like(down_irr_profile)

    solution_A, solution_B, solution_C, solution_D = solutions
    mu = mu[np.newaxis, :]
    ke = ke[:, np.newaxis]
    taustar = taustar[:, np.newaxis]

    direct_only = 0 if streams.compute_2stream_diff else 1

    # it is probably possible to optimize this loop (-> array calculation)
    for nz0, nl, dtaustar_z, taustar_z in depth_generator(dtaustar_ub, taustar):
        expm = np.exp(-ke[nl - 1] * dtaustar_z)
        expp = np.exp(ke[nl - 1] * dtaustar_z)

        down_irr_profile[nz0] = (
            solution_A[nl - 1] * expm
            + solution_B[nl - 1] * expp
            + direct_only * (Gm[nl - 1] + 1) * np.exp(-taustar_z / mu)
        )  # + 1 was mu before
        up_irr_profile[nz0] = (
            solution_C[nl - 1] * expm + solution_D[nl - 1] * expp + direct_only * Gp[nl - 1] * np.exp(-taustar_z / mu)
        )

    return (
        streams.process_output(down_irr_profile, mu, dir_frac),
        streams.process_output(up_irr_profile, mu, dir_frac),
    )


def get_actinic_profile(
    solutions: Sequence[npt.NDArray],
    ke: npt.NDArray,
    dtaustar_ub: ArrayOrScalar,
    taustar: npt.NDArray,
    Gp: npt.NDArray,
    Gm: npt.NDArray,
    mu: npt.NDArray,
    dir_frac: float,
    streams: Streams,
    depth_generator: DepthGenerator,
) -> npt.NDArray[np.floating]:
    # compute the profile of actinic flux
    actinic_profile = np.zeros((len(depth_generator.z), len(mu)))

    solution_A, solution_B, solution_C, solution_D = solutions
    mu = mu[np.newaxis, :]
    ke = ke[:, np.newaxis]
    taustar = taustar[:, np.newaxis]

    diffuse2actinic = np.zeros_like(mu)
    diffuse2actinic[streams.nstreams_dir :] = 1

    # it is probably possible to optimize this loop (-> array calculation)
    for nz0, nl, dtaustar_z, taustar_z in depth_generator(dtaustar_ub, taustar):
        expm = np.exp(-ke[nl - 1] * dtaustar_z)
        expp = np.exp(ke[nl - 1] * dtaustar_z)

        # the factor 2 in the following equations converts the radiation flux into actinic flux.
        # diffuse radiation are given a factor of 2 while the direct radiation (only one term actually) has a factor of 1.
        # this is consistent with the two-stream approximation as used in TARTES.
        actinic_profile[nz0] = (
            2 * (solution_A[nl - 1] + solution_C[nl - 1]) * expm
            + 2 * (solution_B[nl - 1] + solution_D[nl - 1]) * expp
            + (2 * Gm[nl - 1] + 2 * Gp[nl - 1] + 1 / mu + diffuse2actinic) * np.exp(-taustar_z / mu)
        )

    return streams.process_output(actinic_profile, mu, dir_frac)


def estimate_effective_layer_number(ke: npt.NDArray, dtaustar: ArrayOrScalar):
    """estimate the number of layers to take into account at each wavelength

    :param ke: delta-eddington AFEC
    :type ke: array
    :param dtaustar: optical depth of each layer
    :type dtaustar: array

    :returns: number of layers to consider for each wavelength"""
    tau = np.cumsum(ke * dtaustar, axis=1)
    taumax = 30.0  # optical depth from which the absorbed energy is negligible

    nlyrmax = np.empty(ke.shape[0], dtype=np.int32)
    for i in range(len(nlyrmax)):
        # +1 added compared with Quentin's code
        nlyrmax[i] = np.searchsorted(tau[i, :], taumax, side='right') + 1

    return nlyrmax


def soa(x: Union[ArrayOrScalar, dict], i: int) -> float:
    # allow scalar or array/list

    if hasattr(x, '__iter__') and not isinstance(x, dict) and not isinstance(x, str):
        return x[i]
    else:
        return x


def tartes(
    wavelength: ArrayOrScalar,
    SSA: ArrayOrScalar,
    density: Optional[ArrayOrScalar] = None,
    thickness: Optional[ArrayOrScalar] = None,
    shape_parameterization: str = 'robledano23',
    g0: ArrayOrScalar = default_g0,
    B0: Union[str, ArrayOrScalar] = default_B0,
    impurities: ArrayOrScalar = 0.0,
    impurities_type: Type[Impurities] = SootSNICAR3,
    refrac_index: Union[str, ArrayOrScalar, Callable] = 'p2016',
    soilalbedo: ArrayOrScalar = 0.0,
    dir_frac: ArrayOrScalar = 0.0,
    totflux: ArrayOrScalar = 1.0,
    diff_method: str = 'aart eq',
    infprop_method: str = 'delta_eddington',
    mudir: ArrayOrScalar = 0.0,
    return_dir_diff: bool = False,
    compute_absorption: bool = False,
    compute_irradiance_profiles: bool = False,
    compute_actinic_profile: bool = False,
    z: Optional[ArrayOrScalar] = None,
) -> Union[
    npt.NDArray[np.floating],
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]],
]:
    """compute spectral albedo, and optionally the absorption in each layer and in the soil, the downwelling and upwelling
    irradiance profiles and the actinic flux from the physical properties of the snowpack and the incidence flux conditions.

    :param wavelength: wavelength (m)
    :type wavelength: array or scalar
    :param SSA: snow specific surface area (m^2/kg)
    :type SSA: array or scalar
    :param density: snow density (kg/m^3)
    :type density: array or scalar
    :param thickness: thickness of the layers (m)
    :type thickness: array or scalar
    :param shape_parameterization: method to determine the values of B and g as a function of the wavelength.
        "robledano23", the default, use B=n^2 and a constant defaut g. "constant" uses B0 and g0 parameters for all
        wavelengths. "linear" use the linear relationships as a function of the refractive index deduced from Kokhanosvky
        2004 tables (p61).
    :param g0: asymmetry parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit).
        The default value is 0.86. g0 can be scalar (constant in the snowpack) or an array like the SSA.
    :type g0: array or scalar
    :param B0: absorption enhancement parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit).
        The default value is 1.6, taken from Libois et al. 2014. B0 can be scalar (constant in the snowpack) or an array
        like the SSA.
    :type B0: array or scalar
    :param impurities: impurities concentration (g/g/) in each layer. It is either a constant or an array with size
        equal to the number of layers. The array is 1-d if only one type of impurities is used and 2-d otherwise.
    :type impurities: array or scalar
    :param impurities_type: specify the type of impurity. By defaut it is "soot". Otherwise it should be a class (or an
        instance) defining the density and the imaginary part of the refractive index like the Soot class (see
        tartes.impurities for possible choices and to add new impurities types). It can also be a list of classes if several
    impurities type are present in the snowpack. In this case, the impurities parameter must be a 2-d array.
    :type impurities_type: object or list of object
    :param soilalbedo: albedo of the bottom layer (no unit). soilalbedo can be a scalar or an array like wavelength.
    :type soilalbedo: scalar or array
    :param dir_frac: fraction of directional flux over the total flux (the default is dir_frac = 0 meaning 100% diffuse
        incident flux) at every wavelength
    :type dir_frac: array
    :param totflux: total spectral incident flux across the horizontal surface (direct+diffuse). 
        The direct flux can be calculated from the direct beam radiance F0 incoming at the incidence theta_inc with F0 x
        cos(theta_inc). The unit of totflux is at the user's discretion as TARTES internal calculation is relative to
        the incoming flux. Only the returned result is multiplied by totflux. A usual unit is W /m^2 /nm.
    :type totflux: array
    :param mudir: cosine of the incident angle of direct light
    :type mudir: scalar
    :param return_dir_diff: if True return the direct and diffuse components of the albedo. In this case, dir_frac and
        totflux are not used.
    :type return_dir_diff: boolean
    :param diff_method: the method used to compute the diffuse radiation. Possible options are
        1) "aart eq" which perform direct calculation at the 48.2° equivalent angle according to AART,
        2) "integration" which performs the integration of incident radiation over the incidence angle (precise but slow),
        3) "2 streams" which uses diffuse boundary conditions in the 2 stream approximations (inexact for large absorptions,
         not recommended in general).
        The first one is the default, a trade off between speed and accuracy.
    :param compute_absorption: if True compute the absorption profile and the absorption in the soil
    :type compute_absorption: boolean
    :param compute_irradiance_profiles: if True compute the profiles of up- and down-welling irradiance at depths z
    :type compute_actinic_profile: boolean
    :param compute_actinic_profile: if True compute the profile of actinic flux at depths z
    :type compute_irradiance_profiles: boolean
    :param z: depth at which the irradiance is calculed. It is used only if compute_irradiance_profile is activated.
    :type z: array
    :param refrac_index: real and imaginary parts of the refractive index for each wavelength or a string "p2016" for
        Picard et al. 2016 or w2008 for Warren and Brandt 2008 or w1995 for Warren, 1984 modified as noted on Warren
        web site.
    :type refrac_index: string or tuple of two arrays

    :returns: spectral albedo, and optionaly absorption by layer (note the bottom layer correspond to the absorption by
        the soil) and optionnaly the profile of irradiance: downwelling, upwelling."""

    # the diffuse incident flux is treated as a direct flux at an effective incident angle.
    # The value of 53° was used, based on tests with DISORT (Libois 2013).
    # However, from  22 June 2022, it was decided to use a value consistent with the ART theory
    # that is 48° which corresponds to 3/7*(1+2*cos(theta)) = 1.

    if density is None:
        if thickness is not None:
            raise Exception('no density argument is only allowed for semi-infinite snowpacks (thickness=None)')
        density = 300.0  # any value is good for semi-infinite snowpacks

    if thickness is None:
        thickness = 1e9  # chosing a value should never been a problem...
    thickness = np.atleast_1d(thickness)

    nlyr = len(thickness)

    # convert SSA, density and thickness to array if necessary
    if not np.isscalar(SSA):
        SSA = np.array(SSA)
        assert len(SSA) == nlyr

    if not np.isscalar(density):
        density = np.array(density)
        assert len(density) == nlyr

    wavelength = np.atleast_1d(wavelength)
    N = len(wavelength)

    # intermediate variables
    ssalb = np.zeros((N, nlyr))
    g = np.zeros_like(ssalb)

    # 1 compute optical properties for an array of wavelength

    for n in range(nlyr):
        ssalb[:, n], g[:, n] = single_scattering_optical_parameters(
            wavelength,
            refrac_index,
            soa(SSA, n),
            soa(impurities, n),
            impurities_type,
            shape_parameterization,
            soa(g0, n),
            soa(B0, n),
        )

    sigext = density * SSA / 2

    return two_stream_rt(
        thickness,
        ssalb,
        g,
        sigext,
        bottom_albedo=soilalbedo,
        dir_frac=dir_frac,
        diff_method=diff_method,
        infprop_method=infprop_method,
        totflux=totflux,
        mudir=mudir,
        return_dir_diff=return_dir_diff,
        compute_absorption=compute_absorption,
        compute_irradiance_profiles=compute_irradiance_profiles,
        compute_actinic_profile=compute_actinic_profile,
        z=z,
    )


def two_stream_rt(
    thickness: ArrayOrScalar,
    ssalb: npt.NDArray,
    g: npt.NDArray,
    sigext: ArrayOrScalar,
    bottom_albedo: ArrayOrScalar = 0.0,
    dir_frac: ArrayOrScalar = 0.0,
    totflux: ArrayOrScalar = 1.0,
    mudir: ArrayOrScalar = 0,
    return_dir_diff: bool = False,
    diff_method: str = 'aart eq',
    infprop_method: str = 'delta_eddington',
    compute_absorption: bool = False,
    compute_irradiance_profiles: bool = False,
    compute_actinic_profile: bool = False,
    z: Optional[ArrayOrScalar] = None,
) -> Union[
    npt.NDArray,
    tuple[npt.NDArray, npt.NDArray],
    tuple[npt.NDArray, npt.NDArray, npt.NDArray],
]:
    thickness = np.atleast_1d(thickness)

    N = ssalb.shape[0]
    assert N > 0
    nlyr = max(ssalb.shape[1], len(thickness))
    # cosine:

    streams = Streams(
        dir_frac=dir_frac,
        mudir=mudir,
        diff_method=diff_method,
        return_dir_diff=return_dir_diff,
    )

    if streams.compute_2stream_diff and (compute_absorption or compute_irradiance_profiles or compute_actinic_profile):
        raise Exception(
            """compute_absorption_profile, compute_irradiance_profiles and compute_actinic_profile are incompatible 
with the diffuse 2 stream calculation."""
        )

    mu = streams.mu  # in principe streams hold the mu array,
    # but for future implementation in JAX or other AD language it seesm wise to keep mu outside this object.

    albedo = np.empty((N, streams.nstreams_return))

    if compute_absorption:
        energyprofile = np.zeros((N, nlyr + 1, streams.nstreams_return))

    elif compute_irradiance_profiles:
        assert z is not None
        z = np.atleast_1d(z)
        depth_generator = DepthGenerator(z, thickness)
        down_irr_profile = np.empty((N, len(z), streams.nstreams_return))
        up_irr_profile = np.empty_like(down_irr_profile)

    elif compute_actinic_profile:
        assert z is not None
        z = np.atleast_1d(z)
        depth_generator = DepthGenerator(z, thickness)
        actinic_profile = np.empty((N, len(z), streams.nstreams_return))

    # compute infinite layer albedo and ke
    infalb = np.empty_like(ssalb)
    ke = np.empty_like(infalb)

    for n in range(nlyr):
        infalb[:, n], ke[:, n] = infinite_medium_optical_parameters[infprop_method](ssalb[:, n], g[:, n])

    # 2 computation on every wavelength and layer of the optical depth
    dtaustar_ub, dtaustar, taustar = taustar_vector[infprop_method](sigext, thickness, ssalb, g, ke)

    # use to limit the computation at depth for highly absorbing wavelength. Seems to be inefficient in
    # Python for a normal ~40 layers snowpack and 10nm resolution.
    nlyrmax = estimate_effective_layer_number(ke, dtaustar)

    # 3 solve the radiative transfer for each wavelength successively

    # TODO: convert this loop to parallal multi-core computing with joblib
    for i in range(0, N):
        # Number of layer required for the computation
        neff = min(nlyrmax[i], nlyr)
        if compute_irradiance_profiles:
            neff = max(neff, max(depth_generator.nearest_layer) + 1)
        if compute_absorption or compute_actinic_profile:
            neff = nlyr

        layeralbedo_i = infalb[i, :neff]
        ke_i = ke[i, :neff]
        bottom_albedo_i = soa(bottom_albedo, i)
        totflux_i = soa(totflux, i)
        dir_frac_i = soa(dir_frac, i)

        if len(mu) > 0:
            Gp_i, Gm_i = Gp_Gm_vectors[infprop_method](ssalb[i, :neff], ke_i, g[i, :neff], mu)
        else:
            # if only diffuse is required, it is not necessary to compute G
            Gp_i = Gm_i = np.zeros(neff)

        # If the snowpack was truncated, the last layer thickness is increased and a high albedo is used for the soil
        if neff < nlyr:
            dtaustar[i, neff - 1] = 30.0 / ke[i, neff - 1]
            bottom_albedo_i = 1

        taustar_i = taustar[i, :neff]
        dtaustar_i = dtaustar[i, :neff]

        # compute the two-stream matrix
        d = two_stream_matrix(layeralbedo_i, bottom_albedo_i, ke_i, dtaustar_i)

        # compute the vector for direct and diffuse intensities
        vect = two_stream_vector(
            layeralbedo_i,
            bottom_albedo_i,
            dtaustar_i,
            taustar_i,
            Gm_i,
            Gp_i,
            mu,
            streams,
        )

        solutions = solve_two_stream(d, vect, layeralbedo_i)

        # compute the albedo
        albedo[i] = get_albedo(solutions, Gp_i, mu, dir_frac_i, streams)

        if compute_absorption:
            # compute the profile of absorbed energy

            energyprofile[i, :neff] = totflux_i * get_absorption_profile(
                solutions,
                ke_i,
                dtaustar_i,
                taustar_i,
                Gm_i,
                Gp_i,
                mu,
                dir_frac_i,
                streams,
            )
            # compute the energy absorbed by the soil
            energyprofile[i, -1] = totflux_i * get_soil_absorption(
                solutions,
                ke_i,
                dtaustar_i,
                taustar_i,
                Gm_i,
                mu,
                dir_frac_i,
                streams,
                bottom_albedo_i,
            )

        elif compute_irradiance_profiles:
            down, up = get_irradiance_profiles(
                solutions,
                ke_i,
                dtaustar_ub[i],
                taustar_i,
                Gp_i,
                Gm_i,
                mu,
                dir_frac_i,
                streams,
                depth_generator,
            )
            down_irr_profile[i], up_irr_profile[i] = totflux_i * down, totflux_i * up

        elif compute_actinic_profile:
            actinic_profile[i] = totflux_i * get_actinic_profile(
                solutions,
                ke_i,
                dtaustar_ub[i],
                taustar_i,
                Gp_i,
                Gm_i,
                mu,
                dir_frac_i,
                streams,
                depth_generator,
            )

    if not compute_absorption and not compute_irradiance_profiles and not compute_actinic_profile:
        return albedo.squeeze()
    else:
        ret = [albedo.squeeze()]
        if compute_absorption:
            ret.append(energyprofile.squeeze())
        if compute_irradiance_profiles:
            ret.append(down_irr_profile)
            ret.append(up_irr_profile)
        if compute_actinic_profile:
            ret.append(actinic_profile)
        return tuple(ret)
