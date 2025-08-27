# main.py
import argparse
from src.train import train
from src.evaluate  import evaluate
from src.predict import predict

def main():
    parser = argparse.ArgumentParser(description="3D Tumor Segmentation & Volumetry Pipeline")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "evaluate", "predict"],
                        help="Choose the pipeline mode")

    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Path to dataset directory")

    parser.add_argument("--model_path", type=str, default="checkpoints/unet3d.pth",
                        help="Path to save/load the model")

    parser.add_argument("--input_file", type=str, default=None,
                        help="Input NIfTI file for prediction (only in predict mode)")

    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save predictions/visualizations")

    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir, args.model_path)
    elif args.mode == "evaluate":
        evaluate(args.data_dir, args.model_path)
    elif args.mode == "predict":
        if args.input_file is None:
            raise ValueError("Please provide --input_file for prediction mode")
        predict(args.input_file, args.model_path, args.output_dir)
    else:
        raise ValueError("Invalid mode selected. Choose from train, evaluate, predict.")

if __name__ == "__main__":
    main()
