import numpy as np
import matplotlib.pyplot as plt

def plot_vectors(vectors, labels=None, colors=None, xlim=(-5,5), ylim=(-5,5)):
    plt.figure(figsize=(6,6))
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)       
    plt.gca().set_aspect('equal')

    if colors is None:
        colors = ["blue", "red", "green", "orange", "purple"]
        
    for i, v in enumerate(vectors):
        color = colors[i % len(vectors)]
        plt.quiver(0,0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=color)

        if labels and i < len(labels):
            plt.text(v[0]+0.1, v[1]+0.1, labels[i], fontsize=12)

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
        
        
v = [np.array([3, 4])]
plot_vectors(v)


#
# # %%
# # import numpy as np
# # from viz_utils import plot_vectors
#
# v = np.array([3,4])
# plot_vectors([v], labels=["v"])
#
# # # notes run this to import
# # from viz_utils import plot_vectors
# # import numpy as np
#
# # u = np.array([1,2])
# # v = np.array([3,4])
#
# # plot_vectors([u, v], labels=["u","v"])
#
# # # reload
# # from importlib import reload
# # import viz_utils
# # reload(viz_utils)
