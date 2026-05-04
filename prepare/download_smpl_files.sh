mkdir -p body_models
cd body_models/

echo -e "The smpl files will be stored in the 'body_models/smpl/' folder\n"
gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
rm -rf smpl

unzip smpl.zip

echo -e "Converting SMPL_NEUTRAL.pkl to SMPL_NEUTRAL.npz\n"
uv run --with chumpy python ../prepare/convert_smpl_pkl_to_npz.py smpl/SMPL_NEUTRAL.pkl smpl/SMPL_NEUTRAL.npz

echo -e "Cleaning\n"
rm smpl.zip

echo -e "Downloading done!"
