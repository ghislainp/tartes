tartes package
==============

This module provides several functions to compute optical properties of a given snowpack using the TARTES model. The most convenient functions are :py:func:`albedo`, :py:func:`absorption_profile` and :py:func:`irradiance_profiles` to compute respectively the albedo, the profile of absorbed energy and the irradiance profile within the snowpack. These functions are wrapper around the polyvalent function :py:func:`tartes` that performs the computation. The latter function should be used only when a combinaison of albedo, profile of absorbed energy and/or profile of irradiance is needed. 

Module contents
---------------

.. autofunction:: tartes.albedo
.. autofunction:: tartes.ssa

.. autofunction:: tartes.absorption_profile
.. autofunction:: tartes.irradiance_profiles
.. autofunction:: tartes.actinic_profile

.. autofunction:: tartes.tartes


