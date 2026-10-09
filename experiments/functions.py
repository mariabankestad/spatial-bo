"""Alias of spatialbo.functions for the experiment scripts, which import it
as `functions`; the test functions themselves live in the package."""
from spatialbo.functions import *                     # noqa: F401,F403
from spatialbo.functions import (FUNCS, DIM, standardized,                 # noqa: F401
                                 MB4_VARIANTS, MB4_CODIM, MB4_POS, MB4_S, MB4_B)
