import argparse
import model


def main():
    ...


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--name', type=str, default='default')
    args.add_argument('--lr', type=float, default=0.01)
    args.add_argument('--batch_size', type=int, default=32)

    args = args.parse_args()
    main(args)