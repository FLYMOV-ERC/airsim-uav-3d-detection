#!/usr/bin/env python3
"""Extract every image message of one ROS 1 topic from a rosbag to JPEG files.

This is what produced the frame dumps that the rest of this study analyses. The
dumps themselves are not published (several hundred megabytes of pixels); this
script plus the original bag regenerates them, which is what makes leaving them
out safe.

The consecutive-frame MD5 check is kept because a bag that repeats a frame
byte for byte is a recording fault worth knowing about: duplicates are reported
while still being written to disk, so the count of unique frames at the end
flags a stalled camera or a re-published message.

It is *not* a motion check, and neither is ``frame_motion_stats.py``. Both are
blind to a target a few dozen pixels across; reading either as evidence that
nothing flew in front of the camera is the mistake this study made and
retracted (README, "The indoor sequence: the empty detections file is a real
miss"; F20 in ``metrics/findings.csv``). ``locate_moving_target.py`` is the
measurement that can see such a target.

Write each pass to its own ``--out`` directory. Frames are named by message
index, so two passes sharing a directory silently merge into one dump that
looks like a single recording and is not; ``partition_frame_dump.py`` detects
the common case.

Requires ROS 1 (rosbag, cv_bridge). See ``docker/Dockerfile`` for the
environment that was actually used.

Example
-------
    python extract_frames_from_rosbag.py --bag HolybroOut01.bag \
        --topic /camera/color/image_raw --out extracted_images
"""

import argparse
import hashlib
import os
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bag", required=True, help="path to the .bag file")
    p.add_argument(
        "--topic",
        default="/camera/color/image_raw",
        help="image topic to extract (default: /camera/color/image_raw)",
    )
    p.add_argument(
        "--out", required=True, help="output directory for frame_NNNNNN.jpg"
    )
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

    os.makedirs(args.out, exist_ok=True)
    bridge = CvBridge()

    print("Extracting images from bag: %s" % args.bag)
    print("Writing images to:          %s" % args.out)
    print("Using topic:                %s" % args.topic)

    bag = rosbag.Bag(args.bag)

    msg_count = None
    try:
        topics = bag.get_type_and_topic_info()[1]
        if args.topic in topics:
            msg_count = topics[args.topic].message_count
            print("Topic found, %d messages" % msg_count)
        else:
            print("Topic %s not found." % args.topic)
            print("Available topics:")
            for topic in topics.keys():
                print("  - %s" % topic)
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - bag metadata is best-effort
        print("Could not read bag metadata: %s" % exc)

    saved_frames = 0
    saved_unique = 0
    previous_hash = None
    start_time = time.time()

    for idx, (_topic, msg, _t) in enumerate(bag.read_messages(topics=[args.topic])):
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            current_hash = hash_image(cv_image)
            if previous_hash is None:
                if not args.quiet:
                    print("Frame %d: first frame" % idx)
            elif current_hash == previous_hash:
                if not args.quiet:
                    print("Frame %d: byte-identical to the previous one" % idx)
            else:
                if not args.quiet:
                    print("Frame %d: differs from the previous one" % idx)
                saved_unique += 1
            previous_hash = current_hash

            # Every frame is written, duplicates included, under its own name.
            output_path = os.path.join(args.out, "frame_%06d.jpg" % idx)
            if cv2.imwrite(output_path, cv_image):
                saved_frames += 1
                if idx % 10 == 0:
                    print(
                        "Processed %d frames, %d written so far"
                        % (idx, saved_frames)
                    )
            else:
                print("ERROR: failed to write %s" % output_path)

        except Exception as exc:  # noqa: BLE001 - keep going past a bad frame
            print("Error on frame %d: %s" % (idx, exc))
            continue

    bag.close()

    minutes, seconds = divmod(time.time() - start_time, 60)
    print("")
    print("--- Summary ---")
    print("Elapsed:                  %d min %.2f s" % (int(minutes), seconds))
    if msg_count is not None:
        print("Messages on topic:        %d" % msg_count)
    print("Frames written:           %d" % saved_frames)
    print("Frames differing from the previous one: %d" % saved_unique)
    print("Byte-identical frames:    %d" % (saved_frames - saved_unique))
    print("Images written to:        %s" % args.out)
    print("Done.")


if __name__ == "__main__":
    main()
