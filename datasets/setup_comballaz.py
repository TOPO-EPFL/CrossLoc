import argparse
import random
from setup_topodataset_utils import *


def get_ecef_origin():
    """Shift the origin to make the value of coordinates in ECEF smaller and increase training stability"""
    # Warning: this is dataset specific!
    ori_lon, ori_lat, ori_alt = 7.065, 46.3705, 1410
    ori_x, ori_y, ori_z = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978").transform(ori_lat, ori_lon, ori_alt)
    print('Origin XYZ: {}, {}, {}'.format(ori_x, ori_y, ori_z))
    origin = np.array([ori_x, ori_y, ori_z], dtype=np.float64)
    return origin


def main(args):
    """Setup the dataset"""

    print('===== comballaz dataset setup starts =====')

    origin = get_ecef_origin()

    args.lhs_dir = os.path.abspath(os.path.join(args.dataset_dir, args.lhs_dir))

    args.drone_air2_cesium = os.path.abspath(os.path.join(args.dataset_dir, args.drone_air2_cesium))
    args.drone_air2_translated = os.path.abspath(os.path.join(args.dataset_dir, args.drone_air2_translated))
    args.drone_air2_real = os.path.abspath(os.path.join(args.dataset_dir, args.drone_air2_real))

    args.drone_phantom_p_cesium = os.path.abspath(os.path.join(args.dataset_dir, args.drone_phantom_p_cesium))
    args.drone_phantom_p_translated = os.path.abspath(os.path.join(args.dataset_dir, args.drone_phantom_p_translated))
    args.drone_phantom_p_real = os.path.abspath(os.path.join(args.dataset_dir, args.drone_phantom_p_real))

    args.drone_phantom_s_cesium = os.path.abspath(os.path.join(args.dataset_dir, args.drone_phantom_s_cesium))
    args.drone_phantom_s_translated = os.path.abspath(os.path.join(args.dataset_dir, args.drone_phantom_s_translated))
    args.drone_phantom_s_real = os.path.abspath(os.path.join(args.dataset_dir, args.drone_phantom_s_real))
    
    # Check raw data folder paths
    assert os.path.exists(args.lhs_dir)

    assert os.path.exists(args.drone_air2_cesium)
    assert os.path.exists(args.drone_air2_translated)
    assert os.path.exists(args.drone_air2_real)

    assert os.path.exists(args.drone_phantom_p_cesium)
    assert os.path.exists(args.drone_phantom_p_translated)
    assert os.path.exists(args.drone_phantom_p_real)

    assert os.path.exists(args.drone_phantom_s_cesium)
    assert os.path.exists(args.drone_phantom_s_translated)
    assert os.path.exists(args.drone_phantom_s_real)

    dst_dir = os.path.abspath(args.output_dir)
    mkdir(dst_dir)

    # Setup dataset folder by folder
    process_folder(args.lhs_dir, None, dst_dir, 'lhs_sim', origin)  # Cesium synthetic dataset

    # drone data
    drone_data_cesium_ls = [args.drone_air2_cesium, args.drone_phantom_p_cesium, args.drone_phantom_s_cesium]
    drone_data_translated_ls = [args.drone_air2_translated, args.drone_phantom_p_translated, args.drone_phantom_s_translated]
    drone_data_real_ls = [args.drone_air2_real, args.drone_phantom_p_real, args.drone_phantom_s_real]

    for cesium_path, translated_path, real_path in zip(drone_data_cesium_ls, drone_data_translated_ls, drone_data_real_ls):
        process_folder(cesium_path, None, dst_dir, 'drone_sim', origin)
        process_folder(cesium_path, translated_path, dst_dir, 'drone_translated', origin)
        process_folder(cesium_path, real_path, dst_dir, 'drone_real', origin)

    # Split data into training/validation/testing
    for mode in ['lhs_sim', 'drone_sim', 'drone_translated', 'drone_real']:
        print('\n===== Splitting data in {:s} mode ====='.format(mode))
        split_data(os.path.join(dst_dir, mode), mode, 'comballaz')
        shutil.rmtree(os.path.join(dst_dir, mode))

    # Construct virtual dataset pointer for ALL synthetic training data
    src_dir_ls = [os.path.join(dst_dir, 'train_sim'), os.path.join(dst_dir, 'train_drone_sim')]
    virtual_merge_sections(src_dir_ls, os.path.join(dst_dir, 'train_sim_aug'))

    print('===== comballaz dataset setup is done =====')


if __name__ == '__main__':

    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        description='Setup comballaz dataset for single shot visual localization algorithms.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset storing the structured datasets.')

    parser.add_argument('--lhs_dir', type=str, default='comballaz/comballaz-LHS',
                        help='Source directory for synthetic LHS dataset (training set).')

    parser.add_argument('--drone_air2_cesium', type=str,
                        default='comballaz/matching/piloted/djiair2-2020-11-06/comballaz-air2',
                        help='Source directory for Air2 drone equivalent synthetic dataset (test set).')
    parser.add_argument('--drone_air2_translated', type=str,
                        default='comballaz/matching/piloted/djiair2-2020-11-06/comballaz-air2-cyclegantranslatedreal',
                        help='Source directory for translated Air2 drone equivalent synthetic dataset (test set).')
    parser.add_argument('--drone_air2_real', type=str,
                        default='comballaz/matching/piloted/djiair2-2020-11-06/comballaz-air2-real',
                        help='Source directory for real Air2 drone dataset (test set).')

    parser.add_argument('--drone_phantom_p_cesium', type=str,
                        default='comballaz/matching/piloted/phantom4-2020-11-06/comballaz-phantom-piloted',
                        help='Source directory for Phantom piloted equivalent synthetic dataset (test set).')
    parser.add_argument('--drone_phantom_p_translated', type=str,
                        default='comballaz/matching/piloted/phantom4-2020-11-06/comballaz-phantom-piloted-cyclegantranslatedreal',
                        help='Source directory for translated Phantom piloted equivalent synthetic dataset (test set).')
    parser.add_argument('--drone_phantom_p_real', type=str,
                        default='comballaz/matching/piloted/phantom4-2020-11-06/comballaz-phantom-piloted-real',
                        help='Source directory for real Phantom piloted dataset (test set).')

    parser.add_argument('--drone_phantom_s_cesium', type=str,
                        default='comballaz/matching/planned/phantom4-2020-11-06/comballaz-phantom-survey',
                        help='Source directory for Phantom survey equivalent synthetic dataset (test set).')
    parser.add_argument('--drone_phantom_s_translated', type=str,
                        default='comballaz/matching/planned/phantom4-2020-11-06/comballaz-phantom-survey-cyclegantranslatedreal',
                        help='Source directory for translated Phantom survey equivalent synthetic dataset (test set).')
    parser.add_argument('--drone_phantom_s_real', type=str,
                        default='comballaz/matching/planned/phantom4-2020-11-06/comballaz-phantom-survey-real',
                        help='Source directory for real Phantom survey dataset (test set).')

    parser.add_argument('--output_dir', type=str, 
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'comballaz')),
                        help='Destination directory for the organized translated data set. Use DSAC* convention.')

    args = parser.parse_args()

    # fix random seed for reproducibility
    np.random.seed(2021)
    random.seed(2021)

    main(args)
