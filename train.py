from ultralytics import YOLO
from args_ import *
import argparse


def train(args):
    data_path = "/dataset/firstset/data.yaml"
    model = YOLO("/runs/detect/train/weights/best.pt")
    model.train(data=args.dir + data_path, epochs=100, batch=20, workers=8, val=True)
    #model = YOLO(args.dir+"/model/apex_total_8n_350_1000.pt")
    #model.val(data=args.dir + data_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = arg_init(args)
    train(args)
