
'''
Split a batch into real and synthetic batches
returns real, synth batches
'''
def split_batch(
    batch       :dict
):
    real_ids = [i for i,x in enumerate(batch["type"]) if x == "real"]
    synth_ids = [i for i,x in enumerate(batch["type"]) if x == "synthetic"]

    if not real_ids:
        return None, batch, None, synth_ids
    if not synth_ids:
        return batch, None, real_ids, None

    synth_batch = {
        "depth"                 : batch["depth"][synth_ids],
        "normals"               : batch["normals"][synth_ids],
        "labels"                : batch["labels"][synth_ids],
        "color"                 : batch["color"][synth_ids],
        "intrinsics_original"   : batch["intrinsics_original"][synth_ids],
        "intrinsics"            : batch["intrinsics"][synth_ids],
        "camera_resolution"     : batch["camera_resolution"],
        "camera_pose"           : batch["camera_pose"][synth_ids],
        "type"                  : "synthetic"
    }

    real_batch = {
        "depth"                 : batch["depth"][real_ids],
        "intrinsics"             : batch["intrinsics"][real_ids],
        "type"                  : "real"
    }

    return real_batch, synth_batch, real_ids, synth_ids