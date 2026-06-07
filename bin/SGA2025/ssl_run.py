"""
Run ssl_match inference on one or more HDF5 chunks.

Usage:
    python ssl_run.py chunk*.hdf5 --checkpoint resnet50.ckpt --output-dir ./results
"""
import argparse

from SGA.ssl import ssl_match


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ssl_match on HDF5 chunk files.')
    parser.add_argument('hdf5_files', nargs='+', help='HDF5 input file(s)')
    parser.add_argument('--checkpoint', default='resnet50.ckpt',
                        help='Path to the pre-trained resnet50.ckpt checkpoint')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for results (default: same dir as each HDF5 file)')
    parser.add_argument('--no-similarity', dest='similarity', action='store_false',
                        help='Skip similarity-search output')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Similarity threshold (default 0.5)')
    args = parser.parse_args()

    for path in args.hdf5_files:
        ssl_match(path,
                  checkpoint_path=args.checkpoint,
                  output_dir=args.output_dir,
                  similarity=args.similarity,
                  threshold=args.threshold)
