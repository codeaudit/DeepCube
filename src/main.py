from model import build_model
from utils import parse_args
from train import train_model

if __name__ == "__main__":
    args = parse_args
    cost = build_model(args)
    train_model(cost, stream, args)