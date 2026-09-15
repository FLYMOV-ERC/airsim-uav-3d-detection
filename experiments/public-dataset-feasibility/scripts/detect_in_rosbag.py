#!/usr/bin/env python3
"""Run a trained YOLO detector over every image message of a rosbag topic.

This is the script that produced ``results/indoor-transfer/detections.csv``: one
CSV row per detection, plus an annotated JPEG per frame. On the indoor sequence
it produced a header row and nothing else -- zero detections over 948 frames.

That empty file is a detection failure, not an empty recording. An earlier pass
through this material claimed the opposite -- that nothing ever flew in front of
the camera -- on the strength of a whole-frame motion statistic that is
arithmetically blind to a target this small. It is retracted; see the README
section "The indoor sequence: the empty detections file is a real miss" and F20
in ``metrics/findings.csv``. ``locate_moving_target.py`` finds the aircraft
airborne in 447 frames of that same sequence.

Read the result together with ``results/indoor-transfer/indoor-target-track.csv``,
which is the measurement that can see the target, rather than with the
whole-frame CSVs, which cannot.

The weights are not published in this tree (see the README for why); ``--weights``
therefore points at a checkpoint you retrain yourself with ``train_yolov8.py``,
using the configuration recorded in ``results/run-args/``.

One operational warning, because it cost this study a finding. ``--out-images``
is written as ``frame_%06d.jpg`` by message index, so pointing two passes at the
same directory merges them: the shorter run is overwritten only where the
indices overlap, and the tail of the longer one is left behind. That is how
``output_images_holybro2/`` came to hold 1217 Holybro frames followed by 596
frames of an unrelated public dataset. Give every pass its own output
directory, and run ``partition_frame_dump.py`` over any dump whose provenance
you did not personally watch.

Requires ROS 1 (rosbag, cv_bridge). See ``docker/Dockerfile``.

Example
-------
    python detect_in_rosbag.py --bag HolybroStdn01.bag \
        --weights runs/detect/yolov8_holybro3/weights/best.pt \
        --out-images output_images --out-csv detections.csv
"""

import argparse
import csv
import hashlib
import os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bag", required=True, help="path to the .bag file")
    p.add_argument(
        "--topic",
        default="/camera/color/image_raw",
        help="image topic to read (default: /camera/color/image_raw)",
    )
    p.add_argument(
        "--weights", required=True, help="path to the trained YOLO .pt checkpoint"
    )
    p.add_argument(
        "--out-images",
        required=True,
        help="directory for the annotated frame_NNNNNN.jpg files",
    )
    p.add_argument("--out-csv", required=True, help="path of the detections CSV")
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the per-frame duplicate report",
    )
    return p.parse_args()


def hash_image(image):
    return hashlib.md5(image.tobytes()).hexdigest()


def main():
    args = parse_args()

    import cv2
    import rosbag
    from cv_bridge import CvBridge
    from ultralytics import YOLO

    os.makedirs(args.out_images, exist_ok=True)
    bridge = CvBridge()
    model = YOLO(args.weights)

    csv_file = open(args.out_csv, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["frame_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
    )

    bag = rosbag.Bag(args.bag)
    print("Rosbag opened, starting inference.")

    previous_hash = None
    n_detections = 0

    for idx, (_topic, msg, _t) in enumerate(bag.read_messages(topics=[args.topic])):
        if not args.quiet:
            print(
                "Frame %d: seq=%s, stamp=%s"
                % (idx, msg.header.seq, msg.header.stamp)
            )
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            current_hash = hash_image(cv_image)
            if previous_hash is None:
                if not args.quiet:
                    print("Frame %d: first frame recorded." % idx)
            elif current_hash == previous_hash:
                if not args.quiet:
                    print("Frame %d: byte-identical to the previous one." % idx)
            else:
                if not args.quiet:
                    print("Frame %d: differs from the previous one." % idx)
            # Update only after the comparison.
            previous_hash = current_hash

            results = model.predict(cv_image, verbose=False)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = box.cls[0]
                    n_detections += 1

                    csv_writer.writerow(
                        [
                            idx,
                            int(cls),
                            float(conf),
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ]
                    )

                    label = "%s %.2f" % (model.names[int(cls)], conf)
                    cv2.rectangle(
                        cv_image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        cv_image,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            cv2.imwrite(
                os.path.join(args.out_images, "frame_%06d.jpg" % idx), cv_image
            )

            if idx % 10 == 0:
                print("Processed %d frames..." % idx)

        except Exception as exc:  # noqa: BLE001 - keep going past a bad frame
            print("Error on frame %d: %s" % (idx, exc))
            continue

    bag.close()
    csv_file.close()

    print("All images and detections written.")
    print("Total detections: %d -> %s" % (n_detections, args.out_csv))


if __name__ == "__main__":
    main()
