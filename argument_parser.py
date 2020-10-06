import argparse
import os
def parse_arguments(args):
    current_dir = os.path.dirname(__file__)
    usage_text = (
        "Segment inputs."
        "Usage:  python segment_inputs.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    # paths
    parser.add_argument("-i","--input_path", type = str, help = "Path to data to be segmented", required = True)
    parser.add_argument("-o","--output_path", type = str, help = "Path to output directory where predictions will be saved.")
   
    # external data paths
    parser.add_argument("--path_prefix_to_root_folder", type = str, help = "Path to where this file is located")
    parser.add_argument('--device_repository', type = str, help = "Path to device repository", default= "Resources\\data\\device_repository.json")
    parser.add_argument("--path_to_box_obj", type = str, help = "Calibration structure box path (.obj).", default="Resources\\data\\asymmetric_box.obj")
    parser.add_argument("--network_path", type=str, help = 'Path to model params file', default="Resources\\data\\model.onnx")
    parser.add_argument("--path_to_box_ss", type = str, help = "Calibration structure box (subsampled) path (.ply).", default="Resources\\data\\subsampled_asymmetric_structure_corrected.ply")

    # external executables paths
    parser.add_argument("--crf_path", type=str, help = 'Path to crf executable', default="Resources\\executables\\crf\\denseCRF2D.exe")
    parser.add_argument("--g2o_path", type=str, help = 'Path to g2o executable', default="Resources\\executables\\g2o\\cli.exe")

    # parameters regarding input data
    parser.add_argument('--name_pos', type = int, help = "At which position of the filenames is the device name (for id_dename_frameid.pgm is 1)", required=True)
    parser.add_argument("-n","--number_of_pixels", type = int, help = "Number of pixels so that a segmentation is valid.", default = 200)
    parser.add_argument("-s","--scale", default = 0.001, type=float, help = 'Depthmap scale to convert to meters')   
    parser.add_argument("--extension", default = "pgm", type=str, help = 'Filename extension')   
    parser.add_argument("--confidence", type=float, default='0.75', help='What confidence is a valid prediction')
    parser.add_argument("--supervision", type = str, help = "Path to CRF executable.", default = "exr_sup", choices=["color_sup", "exr_sup"])

    # other
    parser.add_argument("--global_iters", type = str, help = " Global (outer - i.e. correspodnence establishment) optimization iterations", default = "5")
    parser.add_argument("--local_iters", type = str, help = "Local (inner) optimization linearization iterations", default = "20")

    return parser.parse_known_args(args)