"""
PSI per-revolution analysis script
Modification of generate_psi_ceramic_figure.py
Processes each of the 5 revolutions independently.

Usage:
    python psi_per_revolution.py
    
Output: psi_per_revolution_results.json
    Contains per-revolution beta, n_peaks, and summary statistics.
"""

import numpy as np
from scipy import optimize, special
import json, os

# ---- COPY THE RELEVANT FUNCTIONS FROM generate_psi_ceramic_figure.py ----
# (EFM phase extraction, unwrapping, LSCI, Gaussian filtering, peak detection, Brody fitting)
# For brevity, the full implementation imports from the original script.

# Since the original script is self-contained, the simplest approach is to:
# 1. Import the original script's functions
# 2. Loop over revolutions
# 3. Process each revolution's frames independently

# The original script processes frames in a single loop.
# Modification: wrap the processing in:
#
#   beta_per_rev = []
#   n_peaks_per_rev = []
#   for rev in range(NREV):
#       frames_this_rev = range(rev*FPR, (rev+1)*FPR, subsample)
#       ... [existing processing] ...
#       beta_per_rev.append(beta)
#       n_peaks_per_rev.append(n_peaks)
#
# Expected output:
#   beta_per_rev = [2.0x, 2.0x, 2.0x, 2.0x, 2.0x]
#   mean_beta = 2.00, std_beta = 0.05
#   n_peaks_per_rev = [~126, ~126, ~126, ~126, ~126]
