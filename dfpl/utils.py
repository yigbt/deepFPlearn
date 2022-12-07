import pathlib
import os
import json

def makePathAbsolute(p: str) -> str:
    path = pathlib.Path(p)
    if path.is_absolute():
        return p
    else:
        return str(path.absolute())


def createDirectory(directory: str):
    path = makePathAbsolute(directory)
    if not os.path.exists(path):
        os.makedirs(path)
def createArgsFromJson(in_json: str, ignore_elements: list, return_json_object: bool):
    arguments = []
    with open(in_json, 'r') as f:
        data = json.load(f)
    for i, j in data.items():
        if str(i) not in ignore_elements:
            i = "--" + str(i)
            j = str(j)
            arguments.append(i)
            arguments.append(j)
    if return_json_object:
        return arguments, data
    return arguments