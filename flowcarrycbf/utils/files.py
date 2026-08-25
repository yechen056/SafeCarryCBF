import flowcarrycbf
from pathlib import Path

# get paths
def get_root_path():
    path = Path(flowcarrycbf.__path__[0]).resolve()
    return path


def get_urdf_path():
    path = get_root_path() / "robots" / "assets" / "urdf"
    return path


def get_usd_path():
    path = get_root_path() / "robots" / "assets" / "usd"
    return path


def get_cfg_path():
    path = get_root_path() / "config"
    return path
