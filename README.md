
TARTES (Two-streAm Radiative TransfEr in Snow model)
========================================================

 TARTES is a fast and easy-to-use optical radiative transfer model used to compute spectral albedo of a given snowpack as well as the profiles of energy absorption, irradiance within the snowpack and actinic flux. TARTES represents the snowpack as a stack of horizontal homogeneous layers. Each layer is characterized by the snow specific surface area (SSA), snow density, impurities amount and type, and two parameters for the geometric grain shape: the asymmetry factor g and the absorption enhancement parameter B. The albedo of the bottom interface can be prescribed. The model is fast and easy to use compared to more elaborated models like DISORT - MIE (Stamnes et al. 1988). It is based on the Kokhanovsky and Zege, (2004) formalism for weakly absorbing media to describe the single scattering properties of each layers and the delta-Eddington approximation to solve the radiative transfer equation. Despite its simplicity, it is accurate in the visible and near-infrared range for pristine snow as well as snow containing impurities represented as Rayleigh scatterers (their size is assumed much smaller than the wavelength) whose refractive indices and concentrations can be prescribed.

TARTES has been initially developed to investigate the influence of the particle shape used to represent snow micro-structure on the penetration of light in the snowpack (Libois et al. 2013, Libois et al. 2014) and to compute the vertical profile of absorbed solar radiation. Nevertheless, it is a general purpose optical radiative transfer model. The latest version 2.0 is described in [Picard and Libois, 2024](https://doi.org/10.5194/gmd-17-8927-2024).

[Documentation](http://gp.snow-physics.science/tartes/)


License 
=========

Copyright (C) 2014, Quentin Libois, Ghislain Picard

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
