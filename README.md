# TransPose: Sim2Real Transfer Learning for Camera Pose Estimation

TransPose: Sim2Real **Trans**fer Learning for Camera **Pose** Estimation

TransPose-NCE: contrastive learning of domain invariant representation

TransPose-MLR: mid-level representation learning for domain invariant robustness 

## Development log

Starting basis: [DSAC*](https://github.com/vislearn/dsacstar)

TO-DO I:

- [x] [BPnP](https://github.com/BoChenYS/BPnP) for efficient end-to-end learning, 
  - [x] Not working and deprecated eventually :(
- [x] Uncertainty learning loss as in [KFNet](https://github.com/zlthinker/KFNet) 
  - [ ] Isotropic covariance is used & things could be improved.
  - [ ] Sampling to get the pixel-wise error & at testing time, we could get multiple coordinate maps and poses.
- [ ] Geometry similarity metrics
  - [x] The idea is similar to the bilateral frustum overlap sampling as in [CamNet](http://openaccess.thecvf.com/content_ICCV_2019/html/Ding_CamNet_Coarse-to-Fine_Retrieval_for_Camera_Re-Localization_ICCV_2019_paper.html)
  - [x] Normalized surface overlap (NSO) sampling as in [Image-box-overlap](https://github.com/nianticlabs/image-box-overlap)
  - [x] Re-ranking of retrieved samples as in [person-re-ranking](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.html)
- [ ] Implement supervised contrastive loss ([SupCon](http://arxiv.org/abs/2004.11362))
  - [ ] To handle unbalanced dataset roll-out use prioritized experience replay as in [PER](https://openreview.net/forum?id=pBbWjZdoRiN)

TO-DO II:

- [ ] Study some visual localization baseline algorithms
  - [ ] Re-implement CycleGAN or CUT for translated image generation [Synthetic-to-real refinement]
  - [ ] Re-implement AtLoc on latest dataset [Absolute pose regression]
  - [ ] Re-implement DSAC* on latest dataset [Scene coordinate regression]
  - [ ] Implement [hloc](https://openaccess.thecvf.com/content_CVPR_2019/html/Sarlin_From_Coarse_to_Fine_Robust_Hierarchical_Localization_at_Large_Scale_CVPR_2019_paper.html) on latest dataset [Descriptor matching]
- [ ] Leverage mid-level representation learning

##  Get started

### Install dependencies

* If `conda` environment is available:

```bash
conda env create -f setup/environment.yml
conda activate TransPose

cd dsacstar && python3 setup_super.py --conda
# sanity check for DSAC* plugin
python3 -c "import torch; import dsacstar; print('DSAC* installation is fine')"
```

Note: `import torch` must be used before `import dsacstar` in the python script. `conda`  environment is preferred as it handles the low-level opencv dependencies quite easily.

* Otherwise, if `conda` environment is not readily available:

```bash
# if at Izar cluster
module load gcc cmake python  
python3 -m venv venvtranspose
source venvtranspose/bin/activate
pip3 install pip -U && pip3 install -r setup/requirements.txt

wget -O opencv-3.4.2.zip https://github.com/opencv/opencv/archive/refs/tags/3.4.2.zip
unzip -q opencv-3.4.2.zip && rm opencv-3.4.2.zip
mkdir -p opencv-build && cd opencv-build
cmake -DCMAKE_INSTALL_PREFIX=install ../opencv-3.4.2
cmake --build . -j12 --target install 
rm -rf ../opencv-3.4.2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/install/lib:$(pwd)/install/lib64
cd ../dsacstar && python3 setup_super.py --cv_path ../opencv-build/install
# sanity check for DSAC* plugin
python3 -c "import torch; import dsacstar; print('DSAC* installation is fine')"
```

Note: 

* Run `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/install/lib:$(pwd)/install/lib64` or equivalent commands in shell to add opencv lib directory **EACH** time before `import dsacstar` .
* If you're at Izar cluster, run `module load gcc cmake` in shell **EACH** time before `import dsacstar`. It's advised to hard-code this in your script.
* Due to an [issue](https://github.com/pytorch/pytorch/issues/57273) in `pytorch 1.9.0` stable release, some redundant warning messages may be popped out in the terminal (`Warning: Leaking Caffe2 thread-pool after fork`). Update to nightly pytorch may solve the issue:


    pip3 uninstall torch torchvision torchaudio
    pip3 install pip -U
    pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
    ```

* For `pytorch3d`, please refer to official installation [tutorial](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). With `conda`, one could install by

  ```bash
  conda install -c bottler nvidiacub cudatoolkit
  git clone https://github.com/facebookresearch/pytorch3d.git
  cd pytorch3d && pip install -e .
  ```
### Download datasets

* Download datasets: we adopt [DSAC*](https://github.com/vislearn/dsacstar) resources and keep their data structure.

```bash
cd datasets

export DATA_DIR=/work/topo/VNAV/Synthetic_Data
echo $DATA_DIR
python setup_epfl.py --dataset_dir $DATA_DIR
python setup_comballaz.py --dataset_dir $DATA_DIR

# public dataset
python setup_7scenes.py
python setup_12scenes.py
python setup_cambridge.py
```

### Training and testing

==To be updated==

* Training (sample command)

```bash
LR=0.0003
ITER=1500000
INITTOL=100
python train_init.py comballaz_lhs_sim output/comballaz_lhs_sim_init.net --learningrate ${LR} --iterations ${ITER} --inittolerance ${INITTOL} --uncertainty
```

* Testing (sample command)

```bash
SEC=test
python test_bpnp.py comballaz_lhs_sim output/comballaz_lhs_sim/bpnp.pth --mode 1 --sparse --section ${SEC} --search_dir --save_map
```

* Visualize uncertainty map (sample command, feel free to visualize other results by changing `keywords` arguments)

```bash
python visualize.py output/comballaz_lhs_sim --search_dir --keywords uncertainty
```

