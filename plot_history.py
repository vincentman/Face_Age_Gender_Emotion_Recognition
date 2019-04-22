import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="This script shows training graph from history file.")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input history h5 file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_path = args.input

    df = pd.read_hdf(input_path, "history")
    input_dir = os.path.dirname(input_path)
    plt.plot(df["output_gender_loss"], label="loss (gender)")
    plt.plot(df["output_age_loss"], label="loss (age)")
    plt.plot(df["output_emotion_loss"], label="loss (emotion)")
    plt.plot(df["val_output_gender_loss"], label="val_loss (gender)")
    plt.plot(df["val_output_age_loss"], label="val_loss (age)")
    plt.plot(df["val_output_emotion_loss"], label="val_loss (emotion)")
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(input_dir, "loss.png"))
    plt.cla()

    plt.plot(df["output_gender_acc"], label="accuracy (gender)")
    plt.plot(df["output_age_myMAE"], label="MAE (age)")
    plt.plot(df["output_emotion_acc"], label="accuracy (emotion)")
    plt.plot(df["val_output_gender_acc"], label="val_accuracy (gender)")
    plt.plot(df["val_output_age_myMAE"], label="val_MAE (age)")
    plt.plot(df["val_output_emotion_acc"], label="val_accuracy (emotion)")
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(os.path.join(input_dir, "accuracy.png"))


if __name__ == '__main__':
    main()
