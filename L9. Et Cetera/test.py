# argparse lib - simpify I/O process
import argparse

# Create the parser and add a description
parser = argparse.ArgumentParser(description="Meow like a cat")

# Add arguments
parser.add_argument("-n", default=1, help="number of times to meow", type=int)
parser.add_argument("-f", default=1, help="volume of meow", type=int)

# Parse arguments
args = parser.parse_args()

# Print "meow" with the specified frequency and volume
for _ in range(args.n):
    print("meow " * args.f)