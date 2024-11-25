import argparse

from trocr_processor import trocr_stream

def main():
    """
    Handles command line arguments and begins the real-time OCR using TrOCR by calling trocr_stream().
    """
    parser = argparse.ArgumentParser()

    # Optional arguments:
    parser.add_argument("-s", "--src", help="Video source for video capture", default=0, type=int)

    args = parser.parse_args()

    # Start the OCR stream
    trocr_stream(source=args.src)

if __name__ == '__main__':
    main()
