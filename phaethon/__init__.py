
# ----- HELIOS ----- #
try:
    # TODO: implement!
    import helios
except ImportError:
    import os
    import sys

    heliopath = os.environ["HELIOPATH"]
    sys.path.append(heliopath)


# ----- MUSPELL ----- #
try:
    import muspell
except ImportError:
    import os
    import sys

    muspellpath = os.environ["MUSPELLPATH"]
    sys.path.append(muspellpath)