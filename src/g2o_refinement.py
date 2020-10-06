import argparse
import os
import sys
import subprocess


def parse_arguments(args):
    usage_text = (
        "Segment inputs."
        "Usage:  python segment_inputs.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    # paths
    parser.add_argument("--poses", type = str, help = "Either a folder where the initial poses are located at (in the form of .extrinsics files) or a structured (.json) file with the named transforms", required = True)
    parser.add_argument("--clouds", type = str, help = " Folder where the initial globally posed point cloud files (*.ply) are located at (each associated with an .extrinsics file)", required = True)
    parser.add_argument("--out", type = str, help = "Folder to save the resulting calibration", required = True)
    parser.add_argument("-e","--executable_path", type = str, help = "Path to graph optimization executable.", required = True)
    parser.add_argument("--reference", type = str, help = "Path to the reference (i.e. identity, fixed alignment anchor) cloud.", required = True)
    parser.add_argument("--log", type = str, help = " Local log file")
    parser.add_argument("--work_dir", type = str, help = " Working directory")
    parser.add_argument("--force_normals", action="store_true", help = "Force (more robust) surface normals calculation for each input cloud")
    parser.add_argument("--save_iters", action="store_true", help = "Save results after each global iteration")
    parser.add_argument("--verbose", action="store_true", help = "Log additional process information apart from functional progress.")
    
    # parameters for g20 optimization
    parser.add_argument("--scale", type = str, help = "Scaling factor to use for scaling the point clouds and the poses' translations to meters", default = "0.001")
    parser.add_argument("--extension", type = str, help = "Extension factor to use for extending the reference's bounding box when cropping the input point clouds", default = "1.1")
    parser.add_argument("--dist_threshold", type = str, help = "Euclidean distance threshold to separate inlier (under it) and outlier (over it) correspondences", default = "0.1")
    parser.add_argument("--dot_threshold", type = str, help = "Cosine distance threshold to separate inlier (over it) and outlier (under it) correspondences", default = "0.7")
    parser.add_argument("--k", type = str, help = "K adjacencies between each point cloud", default = "2")
    parser.add_argument("--global_iters", type = str, help = " Global (outer - i.e. correspodnence establishment) optimization iterations", default = "5")
    parser.add_argument("--local_iters", type = str, help = "Local (inner) optimization linearization iterations", default = "20")
    parser.add_argument("--knn_normals", type = str, help = "Number of neighbors to use for surface orientation (i.e. normal) calculation", default = "10")

    return parser.parse_known_args(args)

def main(argv):
    args, uknown = parse_arguments(argv)

    arguments = [
        " --poses ", args.poses,
        " --clouds ", args.clouds,
        " --out ", args.out,
        " --reference ", args.reference,
        " --scale ", args.scale,
        " --extension ", args.extension,
        " --dist_threshold ", args.dist_threshold,
        " --dot_threshold ", args.dot_threshold,
        " --k ",args.k,
        " --global_iters ",args.global_iters,
        " --knn_normals ", args.knn_normals
    ]

    if args.log:
        arguments += [" --log_file ", args.log]

    if args.work_dir:
        arguments += [" --work_dir ", args.work_dir]

    if args.force_normals:
        arguments.append(" --force_normals ")
    if args.save_iters:
        arguments.append(" --save_iters ")
    if args.verbose:
        arguments.append(" --verbose ")

    arguments.append(" --result " + os.path.join(args.out,"extrinsics.json"))

    subprocess.run([args.executable_path] + [arguments], stdout = sys.stderr, stdin=subprocess.DEVNULL)


if __name__ == "__main__":
    main(sys.argv)