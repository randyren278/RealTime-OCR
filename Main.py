import argparse

from CLIP_processor import clip_stream

def main():
    """
    Handles command line arguments and begins the real-time CLIP object detection by calling clip_stream().
    """
    parser = argparse.ArgumentParser()

    # Optional arguments:
    parser.add_argument("-s", "--src", help="Video source for video capture", default=0, type=int)

    args = parser.parse_args()

    # Start the CLIP object detection stream
    clip_stream(source=args.src)

if __name__ == '__main__':
    main()
