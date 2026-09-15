#!/usr/bin/env python3
"""Plot the horizontal trajectory of a VRPN motion-capture pose topic.

Produces ``figures/holybro-out01-mocap-trajectory.png``. The point of the figure
is scale: the motion-capture volume in which usable 6-DoF ground truth existed
spans 11.53 m by 6.04 m horizontally -- ``measure_plot_extent.py`` measures that
off the rendered figure -- which is an order of magnitude below the target-range
regime the dissertation's detection chapter evaluates.

Requires ROS 1 (rosbag). See ``docker/Dockerfile``.

Example
-------
    python plot_mocap_trajectory.py --bag HolybroOut01.bag \
        --topic /vrpn_client_node/holybro/pose --out trajectory.png
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bag", required=True, help="path to the .bag file")
    p.add_argument(
        "--topic",
        default="/vrpn_client_node/holybro/pose",
        help="PoseStamped topic (default: /vrpn_client_node/holybro/pose)",
    )
    p.add_argument("--out", required=True, help="output PNG path")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--title", default="2-D drone trajectory")
    return p.parse_args()


def main():
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt
    import rosbag

    xs, ys = [], []
    bag = rosbag.Bag(args.bag)
    for _topic, msg, _t in bag.read_messages(topics=[args.topic]):
        xs.append(msg.pose.position.x)
        ys.append(msg.pose.position.y)
    bag.close()

    if not xs:
        raise SystemExit("no messages on topic %s" % args.topic)

    plt.figure(figsize=(6, 5))
    plt.plot(xs, ys, ".-")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(args.title)
    plt.axis("equal")
    plt.savefig(args.out, dpi=args.dpi)

    print("Figure written to %s" % args.out)
    print(
        "Horizontal extent: X %.2f m, Y %.2f m (%d poses)"
        % (max(xs) - min(xs), max(ys) - min(ys), len(xs))
    )


if __name__ == "__main__":
    main()
