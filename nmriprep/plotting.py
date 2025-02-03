import matplotlib.pyplot as plt


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
    plt.imshow(array, cmap='magma', vmax=3000)
    plt.axis('off')
    plt.colorbar()
    plt.title("Radioactivity (uCi/g)")
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
    n = array.shape[-1]
    nrows, ncols = optimal_subplot_grid(n)
    images = []
    fig, axs = plt.subplots(
        nrows,
        ncols,
        gridspec_kw={
            'wspace': 0,
            'hspace': 0
        }
    )
    fig.set_figheight(nrows * 0.85)
    for idx, ax in enumerate(axs.flat):
        if idx < n:
            images.append(
                ax.imshow(
                    # convert to mCi from uCi
                    array[..., idx] / 1000,
                    cmap='magma',
                    vmin=0,
                    vmax=3
                )
            )
        ax.axis('off')
    cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Radioactivity (mCi/g)', fontsize=16)
    fig.savefig(out_name)
    plt.close()
    return
