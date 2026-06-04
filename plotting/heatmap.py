import os

import matplotlib.ticker as ticker
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage


class HeatMap:
    def __init__(self, image, heat_map, gaussian_std=10):
        if isinstance(image, np.ndarray):
            height = image.shape[0]
            width = image.shape[1]
            self.image = image
        else:
            image = Image.open(image)
            width, height = image.size
            self.image = image

        heatmap_array = (np.asarray(heat_map) * 255.0).astype(np.float32)
        heatmap_image = Image.fromarray(heatmap_array)
        heatmap_image_resized = heatmap_image.resize((width, height))
        heatmap_image_resized = ndimage.gaussian_filter(
            np.asarray(heatmap_image_resized),
            sigma=(gaussian_std, gaussian_std),
            order=0,
        )
        self.heat_map = np.asarray(heatmap_image_resized)

    def plot(
        self,
        transparency=0.7,
        color_map="bwr",
        show_axis=False,
        show_original=False,
        show_colorbar=False,
        width_pad=0,
    ):
        if show_original:
            plt.subplot(1, 2, 1)
            if not show_axis:
                plt.axis("off")
            plt.imshow(self.image)
            x, y = 2, 2
        else:
            x, y = 1, 1

        plt.subplot(1, x, y)
        if not show_axis:
            plt.axis("off")
        plt.imshow(self.image)
        plt.imshow(self.heat_map, alpha=transparency, cmap=color_map)
        if show_colorbar:
            plt.colorbar()
        plt.tight_layout(w_pad=width_pad)
        plt.show()

    def save(
        self,
        filename,
        format="png",
        save_path=os.getcwd(),
        transparency=0.7,
        color_map="bwr",
        width_pad=-10,
        show_axis=False,
        show_original=False,
        show_colorbar=False,
        **kwargs,
    ):
        if show_original:
            plt.subplot(1, 2, 1)
            if not show_axis:
                plt.axis("off")
            plt.imshow(self.image)
            x, y = 2, 2
        else:
            x, y = 1, 1

        plt.subplot(1, x, y)
        if not show_axis:
            plt.axis("off")
        plt.imshow(self.image)
        plt.imshow(self.heat_map, alpha=transparency, cmap=color_map)
        if show_colorbar:
            plt.colorbar()
        plt.tight_layout(w_pad=width_pad)
        plt.savefig(
            os.path.join(save_path, filename + "." + format),
            format=format,
            bbox_inches="tight",
            pad_inches=0,
            **kwargs,
        )
        print(f"{filename}.{format} has been successfully saved to {save_path}")


def configure_saliency_plot_style(use_latex=False):
    import seaborn as sns

    sns.set(style="whitegrid", palette="pastel", font_scale=1.2)
    if use_latex:
        plt.rc("text", usetex=True)
    plt.rc("font", family="serif")


def prepare_observation_image(observation):
    image = np.asarray(observation).squeeze()
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0]
    if image.ndim == 3 and image.shape[-1] not in (3, 4):
        return image.mean(axis=-1)
    return image


def format_time_label(index, length, use_latex=False):
    offset = length - 1 - index
    if use_latex:
        return r"$o_{t}$" if offset == 0 else rf"$o_{{t-{offset}}}$"
    return "o_t" if offset == 0 else f"o_t-{offset}"


def plot_saliency_overlay_row(
    saliency_maps,
    observations,
    output_path,
    *,
    alpha=0.5,
    gaussian_std=6,
    cmap="seismic",
    use_latex=False,
):
    configure_saliency_plot_style(use_latex=use_latex)

    length = len(saliency_maps)
    vmin = float(np.min(saliency_maps))
    vmax = float(np.max(saliency_maps))

    fig, axes = plt.subplots(1, length, figsize=(4 * length, 4))
    if length == 1:
        axes = [axes]

    image_artist = None
    for index, axis in enumerate(axes):
        observation_image = prepare_observation_image(observations[index])
        heat_map = HeatMap(
            image=observation_image,
            heat_map=np.asarray(saliency_maps[index]),
            gaussian_std=gaussian_std,
        )

        if np.asarray(heat_map.image).ndim == 2:
            axis.imshow(heat_map.image, cmap="gray")
        else:
            axis.imshow(heat_map.image)

        image_artist = axis.imshow(
            heat_map.heat_map,
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(format_time_label(index, length, use_latex=use_latex), fontsize=24, pad=16)
        axis.axis("off")

    colorbar_axis = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    colorbar = fig.colorbar(image_artist, cax=colorbar_axis, orientation="vertical")
    colorbar.ax.tick_params(labelsize=14)
    colorbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(r"$\mathdefault{%.1e}$"))
    colorbar.update_ticks()

    plt.subplots_adjust(left=0.03, right=0.9, bottom=0.08, top=0.88, wspace=0.05)
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
