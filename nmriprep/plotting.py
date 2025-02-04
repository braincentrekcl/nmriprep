import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_curve(
        std_rad,
        std_gv,
        fitted_gv,
        out_dir,
        out_stem,
        data_rad=None,
        data_gv=None
):
    if data_rad is not None and data_gv is not None:
        plt.scatter(
            data_rad,
            data_gv,
            label="Data values",
            alpha=0.5
        )
    plt.plot(
        std_rad,
        fitted_gv,
        label="Curve fit",
        color="black"
    )
    plt.scatter(
        std_rad,
        std_gv,
        label="Standard values",
        marker="x",
        color="red"
    )
    plt.legend(loc=4)
    plt.xlabel("Radioactivity (uCi/g)")
    plt.ylabel("Inverted image grey values")
    plt.title(f"{out_stem} calibration curve")
    plt.savefig(f"{out_dir / out_stem}_calibration.png")
    plt.close()
    return


def plot_roi(array, out_name, xy, size):
    from matplotlib.patches import Rectangle

    plt.imshow(array, cmap="gray_r")
    plt.gca().add_patch(
        Rectangle(
            xy=xy,
            height=size,
            width=size,
            edgecolor='red',
            fill=False
        )
    )
    plt.axis("off")
    plt.colorbar()
    plt.savefig(out_name)
    plt.close()
    return


def plot_single_slice(array, out_name):
    plt.imshow(array / 1000, cmap='magma', vmax=2)
    plt.axis('off')
    plt.colorbar()
    plt.title("Radioactivity (mCi/g)")
    plt.savefig(out_name)
    plt.close()
    return


def optimal_subplot_grid(n):
    import math

    if n <= 3:
        nrows = 1
        ncols = n
    else:
        # Calculate the square root to get an estimate
        sqrt_n = math.sqrt(n)

        # Determine ncols and nrows based on the square root
        ncols = math.ceil(sqrt_n)
        nrows = math.ceil(n / ncols)

    return nrows, ncols


def plot_mosaic(array, out_name):
    # reshape 3d stack to 2d mosaic
    n = array.shape[-1]
    nrows, ncols = optimal_subplot_grid(n)
    array_padded = np.pad(
        array / 1000,   # convert from uCi to mCi
        ((0, 0), (0, 0), (0, nrows*ncols - n))
    ).transpose((2, 0, 1))
    array_reshaped = array_padded.reshape(
        (nrows,
         ncols,
         array.shape[0],
         array.shape[1])
    )
    mosaic = np.vstack(
        [np.hstack(array_reshaped[i]) for i in range(nrows)]
    )

    # plot mosaic
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    im = ax.imshow(
        mosaic,
        cmap='magma',
        vmax=np.round(
            np.quantile(mosaic[mosaic > 0], 0.99),
            decimals=1
        )
    )
    ax.axis('off')

    # add colorbar
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=ax_cb, orientation="vertical")
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Radioactivity (mCi/g)', fontsize=16)

    fig.savefig(out_name, bbox_inches="tight")
    plt.close()
    return
