"""Thin pulse2percept rendering wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SPVConfig:
    grid_size: int = 32
    rho: int = 300
    axlambda: int = 100
    fps: int = 20
    xrange: tuple[int, int] = (-10, 10)
    yrange: tuple[int, int] = (-10, 10)


def render_video_percepts(input_video: str | Path, output_video: str | Path, config: SPVConfig) -> Path:
    """Render a grayscale video through pulse2percept.

    The dependency is imported lazily so the rest of the package can be used
    without pulse2percept installed.
    """

    import pulse2percept as p2p

    if config.axlambda == 0:
        model = p2p.models.ScoreboardModel(xrange=config.xrange, yrange=config.yrange, rho=config.rho)
    else:
        model = p2p.models.AxonMapModel(
            xrange=config.xrange,
            yrange=config.yrange,
            rho=config.rho,
            axlambda=config.axlambda,
        )
    model.build()

    spacing = 4000 / config.grid_size
    radius = spacing / 5
    grid = p2p.implants.ElectrodeGrid(
        (config.grid_size, config.grid_size),
        spacing,
        etype=p2p.implants.DiskElectrode,
        r=radius,
    )
    implant = p2p.implants.ProsthesisSystem(grid)
    stimulus = p2p.stimuli.VideoStimulus(str(input_video), as_gray=True)
    implant.stim = stimulus.resize(implant.earray.shape)
    percept = model.predict_percept(implant)

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    percept.save(str(output_path), fps=config.fps)
    return output_path

