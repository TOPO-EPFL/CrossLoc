import argparse
import random
from setup_topodataset_utils import *


def get_ecef_origin():
    """Shift the origin to make the value of coordinates in ECEF smaller and increase training stability"""
    # Warning: this is dataset specific!
    ori_lon, ori_lat, ori_alt = 6.5668, 46.5191, 390
    ori_x, ori_y, ori_z = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978").transform(ori_lat, ori_lon, ori_alt)
    print('Origin XYZ: {}, {}, {}'.format(ori_x, ori_y, ori_z))
    origin = np.array([ori_x, ori_y, ori_z], dtype=np.float64)
    return origin


def main(args):
    """Setup the dataset"""

    print('===== EPFL dataset setup starts =====')

    origin = get_ecef_origin()

    args.lhs_dir = os.path.abspath(os.path.join(args.dataset_dir, args.lhs_dir))

    args.piloted_cesium = os.path.abspath(os.path.join(args.dataset_dir, args.piloted_cesium))
    args.piloted_translated = os.path.abspath(os.path.join(args.dataset_dir, args.piloted_translated))
    args.piloted_real = os.path.abspath(os.path.join(args.dataset_dir, args.piloted_real))

    args.planned1_cesium = os.path.abspath(os.path.join(args.dataset_dir, args.planned1_cesium))
    args.planned1_translated = os.path.abspath(os.path.join(args.dataset_dir, args.planned1_translated))
    args.planned1_real = os.path.abspath(os.path.join(args.dataset_dir, args.planned1_real))

    args.planned2_cesium = os.path.abspath(os.path.join(args.dataset_dir, args.planned2_cesium))
    args.planned2_translated = os.path.abspath(os.path.join(args.dataset_dir, args.planned2_translated))
    args.planned2_real = os.path.abspath(os.path.join(args.dataset_dir, args.planned2_real))

    args.planned3_cesium = os.path.abspath(os.path.join(args.dataset_dir, args.planned3_cesium))
    args.planned3_translated = os.path.abspath(os.path.join(args.dataset_dir, args.planned3_translated))
    args.planned3_real = os.path.abspath(os.path.join(args.dataset_dir, args.planned3_real))

    # Check raw data folder paths
    assert os.path.exists(args.lhs_dir)

    assert os.path.exists(args.piloted_cesium)
    assert os.path.exists(args.piloted_translated)
    assert os.path.exists(args.piloted_real)

    assert os.path.exists(args.planned1_cesium)
    assert os.path.exists(args.planned1_translated)
    assert os.path.exists(args.planned1_real)

    assert os.path.exists(args.planned2_cesium)
    assert os.path.exists(args.planned2_translated)
    assert os.path.exists(args.planned2_real)

    assert os.path.exists(args.planned3_cesium)
    assert os.path.exists(args.planned3_translated)
    assert os.path.exists(args.planned3_real)

    dst_dir = os.path.abspath(args.output_dir)
    mkdir(dst_dir)

    # Setup dataset folder by folder
    process_folder(args.lhs_dir, None, dst_dir, 'lhs_sim', origin)  # Cesium synthetic dataset

    # drone data
    drone_data_cesium_ls = [args.piloted_cesium, args.planned1_cesium, args.planned2_cesium, args.planned3_cesium]
    drone_data_translated_ls = [args.piloted_translated, args.planned1_translated, args.planned2_translated, args.planned3_translated]
    drone_data_real_ls = [args.piloted_real, args.planned1_real, args.planned2_real, args.planned3_real]

    for cesium_path, translated_path, real_path in zip(drone_data_cesium_ls, drone_data_translated_ls, drone_data_real_ls):
        process_folder(cesium_path, None, dst_dir, 'drone_sim', origin)
        process_folder(cesium_path, translated_path, dst_dir, 'drone_translated', origin)
        process_folder(cesium_path, real_path, dst_dir, 'drone_real', origin)

    # Split data into training/validation/testing
    for mode in ['lhs_sim', 'drone_sim', 'drone_translated', 'drone_real']:
        print('\n===== Splitting data in {:s} mode ====='.format(mode))
        split_data(os.path.join(dst_dir, mode), mode, 'epfl')
        shutil.rmtree(os.path.join(dst_dir, mode))

    # Construct virtual dataset pointer for ALL synthetic training data
    src_dir_ls = [os.path.join(dst_dir, 'train_sim'), os.path.join(dst_dir, 'train_drone_sim')]
    virtual_merge_sections(src_dir_ls, os.path.join(dst_dir, 'train_sim_aug'))

    print('===== EPFL dataset setup is done =====')


if __name__ == '__main__':

    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        description='Setup EPFL dataset for single shot visual localization algorithms.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset storing the structured datasets.')

    parser.add_argument('--lhs_dir', type=str, default='EPFL/EPFL-LHS',
                        help='Source directory for synthetic LHS dataset.')

    parser.add_argument('--piloted_cesium', type=str,
                        default='EPFL/matching/pilotedflights/2020-09-17/EPFL_2020-09-17-piloted',
                        help='Source directory for piloted drone equivalent synthetic dataset.')
    parser.add_argument('--piloted_translated', type=str,
                        default='EPFL/matching/pilotedflights/2020-09-17/EPFL_2020-09-17-piloted-cyclegantranslatedreal',
                        help='Source directory for translated piloted drone equivalent synthetic dataset.')
    parser.add_argument('--piloted_real', type=str,
                        default='EPFL/matching/pilotedflights/2020-09-17/EPFL_2020-09-17-piloted-real',
                        help='Source directory for real piloted drone dataset.')

    parser.add_argument('--planned1_cesium', type=str,
                        default='EPFL/matching/plannedflights/2020-09-17/EPFL_2020-09-17',
                        help='Source directory for planned equivalent synthetic dataset.')
    parser.add_argument('--planned1_translated', type=str,
                        default='EPFL/matching/plannedflights/2020-09-17/EPFL_2020-09-17-cyclegantranslatedreal',
                        help='Source directory for translated planned equivalent synthetic dataset.')
    parser.add_argument('--planned1_real', type=str,
                        default='EPFL/matching/plannedflights/2020-09-17/EPFL_2020-09-17-real',
                        help='Source directory for real planned dataset.')

    parser.add_argument('--planned2_cesium', type=str,
                        default='EPFL/matching/plannedflights/2020-09-24/EPFL_2020-09-24',
                        help='Source directory for planned equivalent synthetic dataset.')
    parser.add_argument('--planned2_translated', type=str,
                        default='EPFL/matching/plannedflights/2020-09-24/EPFL_2020-09-24-cyclegantranslatedreal',
                        help='Source directory for translated planned equivalent synthetic dataset.')
    parser.add_argument('--planned2_real', type=str,
                        default='EPFL/matching/plannedflights/2020-09-24/EPFL_2020-09-24-real',
                        help='Source directory for real planned dataset.')

    parser.add_argument('--planned3_cesium', type=str,
                        default='EPFL/matching/plannedflights/2020-11-11/EPFL_2020-11-11',
                        help='Source directory for planned equivalent synthetic dataset.')
    parser.add_argument('--planned3_translated', type=str,
                        default='EPFL/matching/plannedflights/2020-11-11/EPFL_2020-11-11-cyclegantranslatedreal',
                        help='Source directory for translated planned equivalent synthetic dataset.')
    parser.add_argument('--planned3_real', type=str,
                        default='EPFL/matching/plannedflights/2020-11-11/EPFL_2020-11-11-real',
                        help='Source directory for real planned dataset.')

    parser.add_argument('--output_dir', type=str,
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'EPFL')),
                        help='Destination directory for the organized translated data set. Use DSAC* convention.')

    args = parser.parse_args()

    # fix random seed for reproducibility
    np.random.seed(2021)
    random.seed(2021)

    main(args)
