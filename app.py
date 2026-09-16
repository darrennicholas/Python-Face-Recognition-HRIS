from __future__ import annotations

import os
import time

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2",
)

os.environ.setdefault(
    "TF_NUM_INTRAOP_THREADS",
    "1",
)

os.environ.setdefault(
    "TF_NUM_INTEROP_THREADS",
    "1",
)

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "KMP_BLOCKTIME",
    "0",
)


from camera_server import (
    HTTP_PORT,
    start_camera_servers_once,
)


CAMERA_URL = (
    f"http://127.0.0.1:"
    f"{HTTP_PORT}/camera_web.html"
)


def main():

    print()
    print(
        "=============================================="
    )
    print(
        "       HR ATTENDANCE FACE RECOGNITION"
    )
    print(
        "=============================================="
    )
    print()


    # ------------------------------------------------------
    # Start services
    # ------------------------------------------------------

    print(
        "🚀 Starting camera services..."
    )


    start_camera_servers_once()


    print()
    print(
        "✅ HR Attendance server READY."
    )


    print(
        "🌐 Camera URL:"
    )


    print(
        CAMERA_URL
    )


    print()
    print(
        "👉 Klik URL di terminal "
        "untuk membuka camera."
    )


    print()
    print(
        "⏳ Waiting for browser connection..."
    )


    print()


    # ------------------------------------------------------
    # Keep application alive
    # ------------------------------------------------------

    try:

        while True:

            time.sleep(3600)

    except KeyboardInterrupt:

        print()
        print(
            "🛑 HR Attendance stopped."
        )


if __name__ == "__main__":

    main()