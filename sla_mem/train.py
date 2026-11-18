"""SLA's training (its scripts/train_align.py) with MEM's video memory, in SLA's env:

    python -m sla_mem.train --memory-cameras observation.images.<a>,observation.images.<b> \\
        [--memory-frames 6] [--memory-stride 25] <SLA config> --exp-name <name> [SLA's flags]

--memory-cameras names the dataset's camera features whose history each sample takes; every
other argument goes to SLA's own command line. PYTHONPATH needs this folder and SLA's
checkout (its scripts/), and the working directory is SLA's training one (its teacher loads
checkpoints/dinov2/ from there). The robot's input transform packs each camera's frames with
sla_mem.mem.as_image().

MEM adds no parameter, so the config's weights can be a memoryless checkpoint's (the
paper's weaker MEM-Posttrain-Only). A launcher of one's own does the same in two lines:
sla_mem.mem.install(frames, stride, cameras), then train_align.main(config).
"""

import argparse
import sys

from sla_mem import mem


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--memory-cameras",
        help="comma-separated dataset camera features, e.g. observation.images.cam_high",
    )
    parser.add_argument(
        "--memory-frames", type=int, default=mem.FRAMES, help="1 is off"
    )
    parser.add_argument(
        "--memory-stride", type=int, default=mem.STRIDE, help="dataset frames"
    )
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if args.memory_frames > 1:
        if not args.memory_cameras:
            parser.error("--memory-cameras is needed with --memory-frames > 1")
        cameras = args.memory_cameras.split(",")
        mem.install(args.memory_frames, args.memory_stride, cameras)
    # Imported here: train_align starts by setting JAX's allocator through the environment.
    from openpi.training import config as _config
    from scripts import train_align

    sys.argv = [sys.argv[0], *rest]
    train_align.main(_config.cli())


if __name__ == "__main__":
    main()
