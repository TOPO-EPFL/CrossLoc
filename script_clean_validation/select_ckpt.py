import os
import glob
import re
import pdb
import argparse
import numpy as np


def _config_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--task', type=str, required=True,
						help='specify the single regression task, should be "coord", "depth", "normal" or "semantics"')

	args = parser.parse_args()
	return args


def _get_iter_num(file_name):
	"""
	Extract number from our specified file name.
	E.g., 'results_ckpt_iter_1108534.net_task_coord.txt' ---> '1108534'
	"""
	iter_num = os.path.basename(file_name).split('.net')[0].split('_')[-1]
	return int(iter_num)


def select_coord_model():
	txt_ls = sorted(glob.glob('./results_*task_coord.txt'))

	median_ls = []
	accuracy_ls = []

	pattern_median = r"Median Error:\s+(\d+.\d+) deg, (\d+.\d+) m"
	pattern_acc_5_5 = r"5m5deg: (\d+.\d+)%"
	pattern_acc_10_7 = r"10m7deg: (\d+.\d+)%"
	pattern_acc_20_10 = r"20m10deg: (\d+.\d+)%"

	with open('results_overall.txt', 'w') as f:
		for this_txt in txt_ls:
			f.write("\nThe results is from {:s}".format(this_txt) + '\n')
			with open(this_txt, 'r') as f_in:
				log = f_in.read()
				for line in f_in.readlines():
					f.write(line)

			if len(re.findall(pattern_median, log)) > 1:
				r = re.findall(pattern_median, log)[-1]

			elif len(re.findall(pattern_median, log)) == 1:
				r = re.findall(pattern_median, log)[0]
			else:
				raise Exception("{:s} is wrong".format(this_txt))
			median_ls.append([float(r[0]), float(r[1])])

			for r in re.findall(pattern_acc_5_5, log):
				acc_5_5 = float(r)

			acc_10_7 = 0.00
			for r in re.findall(pattern_acc_10_7, log):
				acc_10_7 = float(r)

			for r in re.findall(pattern_acc_20_10, log):
				acc_20_10 = float(r)

			accuracy_ls.append([acc_5_5, acc_10_7, acc_20_10])

	median_ls = np.stack(median_ls)  # [N, 2]
	accuracy_ls = np.stack(accuracy_ls)  # [N, 3]
	idx_sort = np.argsort(median_ls[:, 1])  # sort by median positioning error

	with open('results_overall.txt', 'a') as f:
		for rank, idx in enumerate(idx_sort):
			eval_str = "\nRank {:d}, median: {:.2f}m, {:.2f}deg. ".format(rank, median_ls[idx][1], median_ls[idx][0])
			eval_str += "\nAcc5m5deg: {:.1f}%, Acc10m7deg: {:.1f}%, Acc20m10deg: {:.1f}%".format(*accuracy_ls[idx])
			eval_str += "\nfile: {:s}".format(txt_ls[idx])
			if rank < 20:
				print(eval_str)
			f.write(eval_str)

	flag_path = os.path.abspath('FLAG_SELECTED_ITER_{:07d}.nodata'.format(_get_iter_num(txt_ls[idx_sort[0]])))
	with open(flag_path, 'w') as f:
		f.write("This is a dummy file.")
	print("\nCheckpoint selection flag is saved at: {:s}".format(flag_path))


def select_depth_model():
	txt_ls = sorted(glob.glob('./results_*task_depth.txt'))

	median_ls = []
	mean_ls = []

	pattern_abs_rel_err = r"absolute relative error, mean:\s+(\d+.\d+)%, median:\s+(\d+.\d+)%"
	pattern_rms_err_err = r"RMS error, mean:\s+(\d+.\d+)m, median:\s+(\d+.\d+)m"

	with open('results_overall.txt', 'w') as f:
		for this_txt in txt_ls:
			f.write("\nThe results is from {:s}".format(this_txt) + '\n')
			with open(this_txt, 'r') as f_in:
				log = f_in.read()
				for line in f_in.readlines():
					f.write(line)

			for idx, patt in enumerate([pattern_abs_rel_err, pattern_rms_err_err]):
				if len(re.findall(patt, log)) > 1:
					r = re.findall(patt, log)[-1]
				elif len(re.findall(patt, log)) == 1:
					r = re.findall(patt, log)[0]
				else:
					raise Exception("{:s} is wrong".format(this_txt))
				if idx == 0:
					mean_abs_rel = float(r[0])
					median_abs_rel = float(r[1])
				elif idx == 1:
					mean_rms = float(r[0])
					median_rms = float(r[1])
			mean_ls.append([mean_abs_rel, mean_rms])
			median_ls.append([median_abs_rel, median_rms])

	median_ls = np.stack(median_ls)  # [N, 2]
	idx_sort = np.argsort(median_ls[:, 1])  # sort by median RMS error

	with open('results_overall.txt', 'a') as f:
		for rank, idx in enumerate(idx_sort):
			eval_str = "\nRank {:d}, median RMS: {:.2f}m, abs-rel: {:.1f}%. ".format(
				rank, median_ls[idx][1], median_ls[idx][0])
			eval_str += "\nfile: {:s}".format(txt_ls[idx])
			if rank < 20:
				print(eval_str)
			f.write(eval_str)

	flag_path = os.path.abspath('FLAG_SELECTED_ITER_{:07d}.nodata'.format(_get_iter_num(txt_ls[idx_sort[0]])))
	with open(flag_path, 'w') as f:
		f.write("This is a dummy file.")
	print("\nCheckpoint selection flag is saved at: {:s}".format(flag_path))


def select_normal_model():
	txt_ls = sorted(glob.glob('./results_*task_normal.txt'))

	median_ls = []
	mean_ls = []

	pattern_reg_err = r"angular prediction error, mean:\s+(\d+.\d+) deg, median:\s+(\d+.\d+) deg"

	with open('results_overall.txt', 'w') as f:
		for this_txt in txt_ls:
			f.write("\nThe results is from {:s}".format(this_txt) + '\n')
			with open(this_txt, 'r') as f_in:
				log = f_in.read()
				for line in f_in.readlines():
					f.write(line)

			for idx, patt in enumerate([pattern_reg_err]):
				if len(re.findall(patt, log)) > 1:
					r = re.findall(patt, log)[-1]
				elif len(re.findall(patt, log)) == 1:
					r = re.findall(patt, log)[0]
				else:
					raise Exception("{:s} is wrong".format(this_txt))
				mean_err = float(r[0])
				median_err = float(r[1])
			mean_ls.append([mean_err])
			median_ls.append([median_err])

	median_ls = np.stack(median_ls)  # [N, 1]
	idx_sort = np.argsort(median_ls[:, 0])  # sort by median prediction error

	with open('results_overall.txt', 'a') as f:
		for rank, idx in enumerate(idx_sort):
			eval_str = "\nRank {:d}, median prediction error: {:.1f}deg. ".format(
				rank, median_ls[idx][0])
			eval_str += "\nfile: {:s}".format(txt_ls[idx])
			if rank < 20:
				print(eval_str)
			f.write(eval_str)
	flag_path = os.path.abspath('FLAG_SELECTED_ITER_{:07d}.nodata'.format(_get_iter_num(txt_ls[idx_sort[0]])))
	with open(flag_path, 'w') as f:
		f.write("This is a dummy file.")
	print("\nCheckpoint selection flag is saved at: {:s}".format(flag_path))


def select_semantics_model():
	txt_ls = sorted(glob.glob('./results_*task_semantics.txt'))

	median_ls = []
	mean_ls = []

	pattern_acc = r"Pixel accuracy, mean:\s+(\d+.\d+), median:\s+(\d+.\d+)"
	pattern_miou = r"Mean IoU, mean:\s+(\d+.\d+), median:\s+(\d+.\d+)"
	pattern_fwiou = r"Frequency weighted IoU, mean:\s+(\d+.\d+), median:\s+(\d+.\d+)"

	with open('results_overall.txt', 'w') as f:
		for this_txt in txt_ls:
			f.write("\nThe results is from {:s}".format(this_txt) + '\n')
			with open(this_txt, 'r') as f_in:
				log = f_in.read()
				for line in f_in.readlines():
					f.write(line)

			for idx, patt in enumerate([pattern_acc, pattern_miou, pattern_fwiou]):
				if len(re.findall(patt, log)) > 1:
					r = re.findall(patt, log)[-1]
				elif len(re.findall(patt, log)) == 1:
					r = re.findall(patt, log)[0]
				else:
					print(patt)
					raise Exception("{:s} is wrong".format(this_txt))
				if idx == 0:
					mean_acc = float(r[0])
					median_acc = float(r[1])
				elif idx == 1:
					mean_miou = float(r[0])
					median_miou = float(r[1])
				elif idx == 2:
					mean_fwiou = float(r[0])
					median_fwiou = float(r[1])

			mean_ls.append([mean_acc, mean_miou, mean_fwiou])
			median_ls.append([median_acc, median_miou, median_fwiou])

	median_ls = np.stack(median_ls)  # [N, 3]
	idx_sort = np.argsort(median_ls[:, 1])[::-1]  # sort by mean IoU (the higher the better)

	with open('results_overall.txt', 'a') as f:
		for rank, idx in enumerate(idx_sort):
			eval_str = "\nRank {:d}, median FwIOU: {:.2f}, median mIOU: {:.2f}, median accuracy: {:.2f} ".format(
				rank, median_ls[idx][2], median_ls[idx][1], median_ls[idx][0])
			eval_str += "\nfile: {:s}".format(txt_ls[idx])
			if rank < 20:
				print(eval_str)
			f.write(eval_str)

	flag_path = os.path.abspath('FLAG_SELECTED_ITER_{:07d}.nodata'.format(_get_iter_num(txt_ls[idx_sort[0]])))
	with open(flag_path, 'w') as f:
		f.write("This is a dummy file.")
	print("\nCheckpoint selection flag is saved at: {:s}".format(flag_path))


def main():

	args = _config_parser()

	if args.task == 'coord':
		select_coord_model()
	elif args.task == 'depth':
		select_depth_model()
	elif args.task == 'normal':
		select_normal_model()
	elif args.task == 'semantics':
		select_semantics_model()
	else:
		raise NotImplementedError


if __name__ == '__main__':
	main()
