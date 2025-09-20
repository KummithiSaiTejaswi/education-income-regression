import argparse
from train import train_model
from demo import interactive_predict

def main():
    parser = argparse.ArgumentParser(description="Education vs Income Regression Project")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True,
                        help="Choose whether to train a model or predict salary")
    parser.add_argument("--data", type=str, default="data/student_income.csv",
                        help="Path to dataset for training")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.data)
    elif args.mode == "predict":
        interactive_predict()

if __name__ == "__main__":
    main()
