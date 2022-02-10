import pathlib
import os


def makePathAbsolute(p: str) -> str:
    path = pathlib.Path(p)
    if path.is_absolute():
        return p
    else:
        return str(path.absolute())


def createDirectory(directory: str):
    path = makePathAbsolute(pathlib.Path(directory))
    if not os.path.exists(path):
        os.mkdir(path)
