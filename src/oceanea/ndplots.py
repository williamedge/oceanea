import numpy as np
import matplotlib.pyplot as plt
from oceanea.ndvariogram import get_nbins
from wootils import plotnice as pn


def plot_variogram2D(bin_edges, variogram, varcount, labels=['X','Y'],\
                     cmap='magma', vmin=None, vmax=None, hsize=6.5, vsize=6.0, cbar_label='Variance'):
    fig, ax = var2D_template(hsize=hsize, vsize=vsize)
    pc, h1 = var2D_data(fig, ax, bin_edges, variogram, varcount, cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=cbar_label)
    var2D_format(ax, labels=labels)
    return fig, ax


# Plot template dev for 2D variograms
def var2D_template(hsize=6.5, vsize=6.0):
    fig, ax = plt.subplots(2,2, figsize=(hsize, vsize),\
                           gridspec_kw={'width_ratios': [3, 1],\
                                        'height_ratios': [1, 3],\
                                        'hspace': 0.05, 'wspace': 0.05})
    ax_ly = ax[0][0].twinx()
    ax_lx = ax[1][1].twiny()
    ax = np.append(ax, [ax_ly, ax_lx])
    return fig, ax


def var2D_data(fig, ax, bin_edges, variogram, varcount, cmap='magma', vmin=None, vmax=None, cbar_label='Variance'):

    bin_centers = get_nbins(bin_edges)

    h1 = ax[0].scatter(bin_centers[0], variogram[:,0], s=15, c=variogram[:,0],\
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax[3].scatter(variogram[0,:], bin_centers[1], s=15, c=variogram[0,:],\
                    cmap=cmap, vmin=vmin, vmax=vmax)

    # bh = ax[4].bar(bin_centers[0], varcount[:,0], color='grey', alpha=0.5, width=np.diff(bin_edges[0]))
    # ax[5].barh(bin_centers[1], varcount[0,:], color='grey', alpha=0.5, height=np.diff(bin_edges[1]))

    # Main plot - only vectorised method (always need to transpose for pcolormesh)
    pc = ax[2].pcolormesh(bin_centers[0], bin_centers[1], variogram.T, shading='auto',\
                          cmap=cmap, vmin=vmin, vmax=vmax)
    
    cax = ax[0].inset_axes([1.035, 0.47, 0.44, 0.06])
    fig.colorbar(pc, cax=cax, orientation='horizontal', label=cbar_label)
    # fig.legend([h1,bh], ['0-lag marginal','Count'], loc='upper right', bbox_to_anchor=(0.975, 0.895))

    return pc, h1


def var2D_format(ax, labels=['X','Y'], scale=None):
    ax[0].set_ylabel('$\gamma_{}$'.format(labels[0]))
    ax[0].set_xticklabels('')
    ax[0].set_xlim(ax[2].get_xlim())
    ax[0].set_ylim(0, None)
    ax[1].axis('off')
    ax[2].set_ylabel('${}$'.format(labels[1]))
    ax[2].set_xlabel('${}$'.format(labels[0]))
    ax[3].set_xlabel('$\gamma_{}$'.format(labels[1]))
    ax[3].set_yticklabels('')
    ax[3].set_ylim(ax[2].get_ylim())
    ax[3].set_xlim(0, None)
    # ax[4].set_ylabel('${}$-count'.format(labels[0]))
    ax[4].yaxis.set_ticks_position('none') 
    ax[4].set_yticklabels('')
    # ax[4].set_ylim(ax[0].get_ylim())
    # ax[5].set_xlabel('${}$-count'.format(labels[1]))
    ax[5].set_xticklabels('')
    ax[5].xaxis.set_ticks_position('none')
    # ax[5].set_xlim(ax[3].get_xlim())

    # Bar patch stuff
    ax[0].patch.set_alpha(0.0)
    ax[0].set_zorder(2)
    ax[4].set_zorder(1)

    if scale is not None:
        ax[2].set_xscale(scale)
        ax[2].set_yscale(scale)
        ax[4].set_aspect(scale)
        ax[5].set_aspect(scale)

    pn.plot_align([ax[0], ax[2]])