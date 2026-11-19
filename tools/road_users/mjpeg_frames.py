#!/usr/bin/env python3
"""mjpeg_frames.py RIDE.mjpeg OUT_DIR [--every N]: split a ride library MJPEG (roadsense docs/RIDES.md: 20 fps,
JPEG back to back) into OUT_DIR/<frame index>.jpg, every N-th frame (default 4: 5 a second, the rate of the
phone's light detector). Frame index i has the time i * 50 ms of the ride's road file (road seq = i + 1)."""
import os, sys


def main():
    src, out = sys.argv[1], sys.argv[2]
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 4
    os.makedirs(out, exist_ok=True)
    buf, i, n = b"", 0, 0
    with open(src, "rb") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            buf += chunk
            while True:
                a = buf.find(b"\xff\xd8")
                if a < 0:
                    buf = b""
                    break
                b = buf.find(b"\xff\xd9", a + 2)
                if b < 0:
                    buf = buf[a:]
                    break
                if i % every == 0:
                    with open(os.path.join(out, "%d.jpg" % i), "wb") as g:
                        g.write(buf[a:b + 2])
                    n += 1
                i += 1
                buf = buf[b + 2:]
    print("frames", i, "written", n, file=sys.stderr)


if __name__ == "__main__":
    main()
