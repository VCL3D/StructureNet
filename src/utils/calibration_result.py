import json
import numpy
import datetime

class CalibrationResult:
    def __init__(
        self,
        method = "structureNet"
        ):
        self.data = {}
        self.data["Metadata"] = {
            "Method"    : method,
            "Date"      : str(datetime.datetime.now())
        }
        self.data["Viewpoints"] = []

    def update(
        self,
        name            :str,
        extrinsics      :numpy.array,
        intrinsics      :numpy.array,
        correspondences :numpy.array
    ):
        self.data["Viewpoints"].append({
            "name"              : name,
            "extrinsics"        : extrinsics.flatten().tolist(),
            "intrinsics"        : intrinsics.flatten().tolist()
        })

    def write(
        self,
        filename        : str
    ):
        if ".json" not in filename:
            filename += ".json"
        with open(filename, 'w') as outfile:
            json.dump(self.data, outfile, indent = 4)
        outfile.close()

    def read(self, filename : str) :

        with open(filename,'r') as infile:
            self.data = json.load(infile)