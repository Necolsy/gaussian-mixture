import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def ellipses(ax, mean, cov, confidence=5.991, alpha=0.3, color="blue"):
    lambda_, v = np.linalg.eig(cov) 
    sqrt_lambda = np.sqrt(np.abs(lambda_))  

    s = confidence
    width = 2 * np.sqrt(s) * sqrt_lambda[0]  
    height = 2 * np.sqrt(s) * sqrt_lambda[1]  
    angle = np.rad2deg(np.arccos(v[0, 0]))  
    ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, color=color)    # 绘制椭圆

    ax.add_artist(ell)
    ell.set_alpha(alpha)
