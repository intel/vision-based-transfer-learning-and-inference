import splitfolders
import argparse
import sys

def split_folder(in_dir,out_dir):
    """
    Splitting the folder into train and test files in the ratio 80:20 
    """
    splitfolders.ratio(in_dir, output=out_dir,
      seed=100, ratio=(.8, .2), group_prefix=None, move=False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--INDIR", help="Specify the directory where unzipped dataset is present",  type=str)
    parser.add_argument("--OUTDIR", help="Specify directory where you want the dataset to be split",  type=str)
    args = parser.parse_args()
    if (len(sys.argv)) != 3:
        print(len(sys.argv))
        parser.print_help()
        sys.exit()
    else:
        split_folder(args.INDIR,args.OUTDIR)
        print("Done")