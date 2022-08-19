import random
import argparse
from setup_topodataset_utils import *


def _reset_seed():
    """Fix random seed for reproducibility"""
    np.random.seed(2021)
    random.seed(2021)


def main():
    """Split sectional dataset reproducibly"""

    args = config_parser()
    assert os.path.exists(args.section_dir), 'section_dir {:s} does not exist!'.format(args.section_dir)
    assert os.path.isdir(args.section_dir), 'section_dir {:s} is not a directory!'.format(args.section_dir)
    assert isinstance(args.chunk_size, list) and len(args.chunk_size), 'chunk_size {} is not a non-empty list'.format(args.chunk_size)
    args.section_dir = os.path.abspath(args.section_dir)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.section_dir)

    print('===== Section split starts =====')
    dp_names = os.listdir(os.path.join(args.section_dir, 'rgb'))
    dp_names = sorted([item.split('.')[0] for item in dp_names])  # remove extention name
    print("Original section at {:s} has {:d} data points.".format(args.section_dir, len(dp_names)))

    # reproducible shuffle
    _reset_seed()
    dp_names = np.random.permutation(dp_names)

    # cumulative sub-sections
    for chunk in sorted(args.chunk_size):
        subset_size = int(np.ceil(chunk * len(dp_names)))
        out_dir = os.path.join(args.output_dir, os.path.basename(args.section_dir) + '_chunk_{:.2f}'.format(chunk))
        print("Split {:.0f}% ({:d} / {:d}) data into {:s}".format(chunk * 100, subset_size, len(dp_names), out_dir))
        virtual_split_section(args.section_dir, out_dir, permitted_items=dp_names[:subset_size])

    print('===== Section split is done =====')


def virtual_split_section(src_dir, dst_dir, permitted_items=None):
    """
    Construct virtual dataset pointer for sub-section
    """
    mkdir(dst_dir)
    for sub_dir in ['calibration', 'init', 'poses', 'rgb', 'depth', 'normal']:
        mkdir(os.path.join(dst_dir, sub_dir))
        paths_from_ls = sorted(glob.glob(os.path.join(src_dir, sub_dir, '*')))
        paths_from_ls = [os.path.abspath(item) for item in paths_from_ls]
        for path_from in paths_from_ls:
            flag_proceed = False
            if permitted_items is None:
                flag_proceed = True
            else:
                dp_name = os.path.basename(path_from).split('.')[0]  # remove extension name
                if dp_name in permitted_items:
                    flag_proceed = True

            if flag_proceed:
                path_to = os.path.abspath(os.path.join(dst_dir, sub_dir, os.path.basename(path_from)))
                if os.path.exists(path_to):
                    os.remove(path_to)
                os.symlink(path_from, path_to)
            else:
                continue


def config_parser():
    """Configure parser"""
    parser = argparse.ArgumentParser(description="Split a section into different chunks in a cumulative way",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--section_dir', type=str, required=True,
                        help="Directory of the sectional data to split")

    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory to save the chunk data (symbolic link pointer)")

    parser.add_argument('--chunk_size', type=float, nargs='+', required=True,
                        help="Proportion of the data chunk after splitting")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    main()
