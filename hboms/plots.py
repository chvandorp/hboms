import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from typing import Literal

from .observation import Observation
from .state import StateVar


def plot_fits(
    sams: dict,
    data: dict,
    state: list[StateVar],
    obs: list[Observation],
    yscale: Literal["linear", "log"] = "linear",
    ylim: tuple[float, float] | None = None,
    ppd: Literal["bar", "envelope"] | None = "bar",
) -> plt.figure:
    # FIXME: plot vector-valued data and states
    # generate a list of default colors
    colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    num_colors = len(colors)

    Time = data["Time"]
    R = len(Time)

    nrows = int(np.sqrt(R))
    ncols = R // nrows + (0 if R % nrows == 0 else 1)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(3 * ncols, 2 * nrows), sharex=True, sharey=True
    )

    # make a list of observations and their shape
    # scalars have shape ()
    indexed_obs = []
    for ob in obs:
        ob_shape = sams[f"{ob.name}_sim"].shape[3:]
        # NB: remove first 3 dimensions: chain, unit, time
        for index, x in np.ndenumerate(np.zeros(ob_shape)):
            indexed_obs.append((ob, index))

    for r, ax in enumerate(axs.flatten()):
        # remove unused axes
        if r >= R:
            ax.axis("off")
            continue
        # plot data and ppds
        if "ID" in data:
            ax.set_title(data["ID"][r])
        for j, (ob, idx) in enumerate(indexed_obs):
            color = colors[j % num_colors]
            ts = Time[r]
            xs = np.array(data[ob.name][r])[(slice(None), *idx)]
            if ob.censored:
                # highlight censored observations
                cs = data[ob.cc_name][r]
                for cc, m in zip([0, -1, 1], ['o', 'v', '^']):
                    txs = [(t, x) for t, x, c in zip(ts, xs, cs) if c == cc]
                    ax.scatter(
                        [t for t, x in txs], [x for t, x in txs], 
                        color=color, zorder=2, s=20, marker=m, linewidths=0
                    )
            else:
                ax.scatter(ts, xs, s=20, color=color, zorder=2, linewidths=0)
            Xs = sams[f"{ob.name}_sim"][(slice(None), r, slice(None), *idx)]
            lXs, mXs, uXs = np.percentile(Xs, axis=0, q=[2.5, 50, 97.5])
            match ppd:
                case "bar":
                    for t, l, u in zip(ts, lXs, uXs):
                        ax.plot([t, t], [l, u], linewidth=1, color=color, zorder=2)
                case "envelope":
                    ax.fill_between(
                        ts, lXs, uXs, linewidth=0, color=color, alpha=0.2, zorder=0
                    )
                case _:
                    pass

        # make a list of state vars and their shape
        # scalars have shape ()
        indexed_state = []
        for var in state:
            var_shape = sams[f"{var.name}_sim"].shape[3:]
            # NB: remove first 3 dimensions: chain, unit, time
            for index, x in np.ndenumerate(np.zeros(var_shape)):
                indexed_state.append((var, index))

        # plot fit
        for j, (var, idx) in enumerate(indexed_state):
            color = colors[j % num_colors]
            traj = sams[f"{var.name}_sim"][(slice(None), r, slice(None), *idx)]
            ts = Time[r]
            ts = np.linspace(ts[0], ts[-1], traj.shape[1])
            lx, mx, ux = np.percentile(traj, axis=0, q=[2.5, 50, 97.5])
            ax.plot(ts, mx, color=color, zorder=1)
            ax.fill_between(ts, lx, ux, alpha=0.4, color=color, zorder=1, linewidth=0)
        # set scale
        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(*ylim)

    return fig
