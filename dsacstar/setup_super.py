"""
Based on the DSAC* code.
https://github.com/vislearn/dsacstar/tree/master/dsacstar

Copyright (c) 2020, Heidelberg University
Copyright (c) 2021, EPFL
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
import argparse
import subprocess


def main():
    """
    Setup the dsac* utility properly.
    """

    opt = config_parser()

    assert opt.conda or opt.cv_path is not None, 'Please specify python environmental variable!'

    if opt.conda:
        raw_setup_path = os.path.join(os.path.dirname(__file__), 'setup.py')
        subprocess.run('python3 {:s} clean'.format(raw_setup_path), shell=True)
        subprocess.run('python3 {:s} install'.format(raw_setup_path), shell=True)
    else:
        opt.cv_path = os.path.abspath(opt.cv_path)
        assert os.path.exists(opt.cv_path)

        opencv_inc_dir = os.path.abspath(os.path.join(opt.cv_path, 'include'))
        if os.path.exists(os.path.join(opt.cv_path, 'lib64')):
            opencv_lib_dir = os.path.join(opt.cv_path, 'lib64')
        else:
            opencv_lib_dir = os.path.join(opt.cv_path, 'lib')
        assert os.path.exists(opencv_inc_dir) and os.path.exists(opencv_lib_dir), "OpenCV 3 build is not complete!"

        # modify the temporary setup script to point to the correct path
        subprocess.run('cp setup.py setup_tmp.py', shell=True)
        for mode, sed_path in zip(['inc', 'lib', 'conda_env'], [opencv_inc_dir, opencv_lib_dir, opt.cv_path]):
            sed_path = '"' + sed_path + '"'
            for sp_mark in ['$', '*', '.', '[', '/', '^']:
                sed_path = sed_path.replace(sp_mark, '\\'+sp_mark)
            if mode == 'conda_env':
                sed_cmd = "sed -i 's/conda_env = os.*/conda_env = {:s}/g' setup_tmp.py".format(sed_path)
            else:
                sed_cmd = "sed -i 's/opencv_{:s}_dir = ''.*/opencv_{:s}_dir = {:s}/g' setup_tmp.py".format(
                    mode, mode, sed_path)
            subprocess.run(sed_cmd, shell=True)

        # installation
        tmp_setup_path = os.path.join(os.path.dirname(__file__), 'setup_tmp.py')
        subprocess.run('python3 {:s} clean'.format(tmp_setup_path), shell=True)
        subprocess.run('python3 {:s} build'.format(tmp_setup_path), shell=True)
        subprocess.run('python3 {:s} install'.format(tmp_setup_path), shell=True)
        subprocess.run('rm setup_tmp.py', shell=True)

    print("DSAC* is successfully installed!")


def config_parser():
    """Configure the argument parser"""
    parser = argparse.ArgumentParser(description="Setup the DSAC* utility.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--conda', action='store_true',
                        help='to search for opencv3 build in the anaconda environment.')

    parser.add_argument('--cv_path', type=str, default=None,
                        help='path to the opencv3 build.')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    main()
