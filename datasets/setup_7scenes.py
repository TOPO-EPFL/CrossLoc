"""
7scenes dataset setup script taken from DSAC* (identical)
https://github.com/vislearn/dsacstar/blob/master/datasets/setup_7scenes.py

Copyright (c) 2020, Heidelberg University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os

# name of the folder where we download the original 7scenes dataset to
# we restructure the dataset by creating symbolic links to that folder
src_folder = '7scenes_source'
focallength = 525.0

def mkdir(directory):
	"""Checks whether the directory exists and creates it if necessacy."""
	if not os.path.exists(directory):
		os.makedirs(directory)

# download the original 7 scenes dataset for poses and images
mkdir(src_folder)
os.chdir(src_folder)

for ds in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:
	
	print("=== Downloading 7scenes Data:", ds, "===============================")

	os.system('wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/' + ds + '.zip')
	os.system('unzip ' + ds + '.zip')
	os.system('rm ' + ds + '.zip')
	
	sequences = os.listdir(ds)

	for file in sequences:
		if file.endswith('.zip'):

			print("Unpacking", file)
			os.system('unzip ' + ds + '/' + file + ' -d ' + ds)
			os.system('rm ' + ds + '/' + file)

	print("Linking files...")

	target_folder = '../7scenes_' + ds + '/'

	def link_frames(split_file, variant):

		# create subfolders
		mkdir(target_folder + variant + '/rgb/')
		mkdir(target_folder + variant + '/poses/')
		mkdir(target_folder + variant + '/calibration/')

		# read the split file
		with open(ds + '/' + split_file, 'r') as f:
			split = f.readlines()	
		# map sequences to folder names
		split = ['seq-' + s.strip()[8:].zfill(2) for s in split]

		for seq in split:
			files = os.listdir(ds + '/' + seq)

			# link images
			images = [f for f in files if f.endswith('color.png')]
			for img in images:
				os.system('ln -s ../../../'+src_folder+'/'+ds+'/'+seq+'/'+img+ ' ' +target_folder+variant+'/rgb/'+seq+'-'+img)

			# link folders
			poses = [f for f in files if f.endswith('pose.txt')]
			for pose in poses:
				os.system('ln -s ../../../'+src_folder+'/'+ds+'/'+seq+'/'+pose+ ' ' +target_folder+variant+'/poses/'+seq+'-'+pose)
			
			# create calibration files
			for i in range(len(images)):
				with open(target_folder+variant+'/calibration/%s-frame-%s.calibration.txt' % (seq, str(i).zfill(6)), 'w') as f:
					f.write(str(focallength))
	
	link_frames('TrainSplit.txt', 'train')
	link_frames('TestSplit.txt', 'test')
