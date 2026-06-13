#!/usr/bin/env python3
"""
Convert existing PNG figures to PDF (vector container with embedded raster).
For true vector regeneration, re-run the source notebooks with PDF backend.
At 300 DPI these are indistinguishable from true vector for review purposes.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

png_files = [
    'fig_d2_scaling.png',
    'fig_multifractal.png', 
    'fig_psd_scaleup.png',
    'fig_dq_spectra.png',
    'fig_convergence.png',
]

for png_name in png_files:
    if not os.path.exists(png_name):
        print(f"⚠ {png_name} not found, skipping")
        continue
    
    # Read PNG at native resolution
    img = mpimg.imread(png_name)
    h_px, w_px = img.shape[:2]
    
    # Use the original DPI (200) to compute figure size
    dpi = 200
    fig_w = w_px / dpi
    fig_h = h_px / dpi
    
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation='none')  # preserve pixel boundaries
    ax.axis('off')
    
    pdf_name = png_name.replace('.png', '.pdf')
    fig.savefig(pdf_name, dpi=dpi, format='pdf', bbox_inches='tight',
                pad_inches=0, facecolor='white', edgecolor='none')
    plt.close(fig)
    
    size_kb = os.path.getsize(pdf_name) / 1024
    print(f"✅ {png_name} → {pdf_name} ({size_kb:.0f} KB)")

print("\nDone. All PNGs converted to PDF.")
print("NOTE: For camera-ready submission, regenerate from source with matplotlib PDF backend for true vector output.")
