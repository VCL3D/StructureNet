try:
    import torch
    import sys
    import os
    import argparse
    import errno
    import warnings
    import onnxruntime

    import importlib.util

    def parse_arguments(args):
        usage_text = (
            "ONNX converter."
            "Usage:  python convert_model_to_onnx.py [options],"
            "   with [options]:"
        )
        parser = argparse.ArgumentParser(description=usage_text)
        # paths
        parser.add_argument("-n","--network_path", type=str, help = 'Path to model params file', required = True)
        parser.add_argument("--structurenet_module", type=str, help = 'Path to structurenet module', required = True)
        parser.add_argument("-o","--output_path", type = str, help = "Path to output directory where the network will be saved.", required = True)
        parser.add_argument("--name", type = str, help = "Name of the network (for saving).", default = "model")
        return parser.parse_known_args(args)


    def main(argv):
        args, uknown = parse_arguments(argv)

        if not os.path.exists(args.network_path):
            raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), args.network_path)

        if not os.path.exists(args.structurenet_module):
            raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), args.structurenet_module)

        if not os.path.exists(args.output_path):
            warnings.warn(args.network_impl + " does not exist, will create.")
            os.makedirs(args.output_path)
        
        sys.path.append(args.structurenet_module)
        import models

        checkpoint = torch.load(args.network_path)
        model_params = {
            'width': 320,
            'height': 180,
            'ndf': 32,
            'upsample_type': "nearest",        
        }
        model_params['ndf'] = checkpoint['ndf']
        model_params['nclasses'] = checkpoint['nclasses']
        model = models.get_UNet_model(checkpoint["model_name"], model_params)
        model.load_state_dict(checkpoint['state_dict'])

        x = torch.randn((1,1,model_params["height"], model_params["width"]))
        y = model(x)[1]
        model_path = os.path.join(args.output_path, args.name + ".onnx")
        torch.onnx.export(model, x, model_path, input_names = "input")

        sess = onnxruntime.InferenceSession(model_path)
        out_ort = torch.tensor(sess.run(None, {"i" : x.numpy()})[1])
        diff = (y[out_ort != y] - out_ort[out_ort != y]).abs().max()
        print("Maximum difference in predictions {}".format(diff))

    if __name__ == "__main__":
        main(sys.argv)
except:
    pass