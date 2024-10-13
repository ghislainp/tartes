
import numpy as np
from tartes.impurities import SootBond06

def test_soot():

    cabs = SootBond06.absorption_crosssection_efficiency(900e-9)
    np.testing.assert_allclose(cabs, 0.2545692778211603)

