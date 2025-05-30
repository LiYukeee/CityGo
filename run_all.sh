## Prerequisites: A fully trained GS for the entire scene, as well as data files for separating Residual and Ambient GS.
# &>> ./run.txt
# Source Path and Output Path
SOURCE_PATH="/lv01/dataset/jingan_high/colmap_ud"  # Path to Colmap dataset
BLOCKS_PATH="/lv01/dataset/jingan_high"  # Path to the directory containing bounding box files
TEST_PATH="/lv01/dataset/jingan_high/block_7"  # Test on a block in central area
OUTPUT_PATH="outputs/JinganHigh"  # Output Path
SPLIT_FILE_PATH="toolkit/data/jingan-high"  # According to the files here, separate the residual and ambient components.
DATA_DEVICE="disk"

# Base MESH and 3DGS point Path
BASE_3DGS_POINT_PATH="/lv01/dataset/jingan_high/others/3dgs/3dgs.ply"
BASE_MESH_PATH="/lv01/dataset/jingan_high/others/mesh/update_0.25uv"

# Iteration Define
RESIDUAL_ITERATION=100000
AMBIENT_ITERATION=30000

# IF Eval
IF_EVAL=true
if [ "$IF_EVAL" == "true" ]; then
    eval="--eval"
else
    eval=""
fi


echo "================================================="
echo "-------------------- STAGE 1 --------------------"
echo "---------- Stage 1.1 - Split Residual and Ambient GS. ----------"
# This Command will create two files:
# $OUTPUT_PATH/temp-file/stage1-residual.ply
# $OUTPUT_PATH/temp-file/stage1-ambient.ply
mkdir -p $OUTPUT_PATH/temp-file/
python toolkit/split_residual_ambient.py \
    --DATA_DIR $SPLIT_FILE_PATH \
    --PLY_PATH $BASE_3DGS_POINT_PATH \
    --OUTPUT_PREFIX $OUTPUT_PATH/temp-file/stage1 \

echo "================================================="
echo "---------- Stage 1.2 - Calculating GS score. ----------"
python calcuate_gs_score.py \
    $eval \
    --data_device $DATA_DEVICE \
    -s $SOURCE_PATH \
    -m $OUTPUT_PATH \
    --mesh_path $BASE_MESH_PATH \
    --ply_path $OUTPUT_PATH/temp-file/stage1-residual.ply \
    --output_path_gs_score $OUTPUT_PATH/temp-file/stage1-GS_SCORE.pt

# Sparse GS point according to GS SCORE
echo "================================================="
echo "---------- Stage 1.3 - Sparse GS point according to GS SCORE. ----------"
python toolkit/choose_gs_accord_score.py \
    --ply_path $OUTPUT_PATH/temp-file/stage1-residual.ply \
    --gs_score_path $OUTPUT_PATH/temp-file/stage1-GS_SCORE.pt \
    --threshold 0.2 \
    --output $OUTPUT_PATH/temp-file/stage1-sparse-residual-GS.ply

# DownSample Ambient GS based on Important Score
echo "================================================="
echo "---------- Stage 1.4 - DownSample Ambient GS based on Important Score. ----------"
echo "---------- XXXXX CODE NEED TO BE COMPLETED. ----------"
python toolkit/random_sample.py \
    --PLY_PATH $OUTPUT_PATH/temp-file/stage1-ambient.ply \
    --OUTPUT_PLY_PATH $OUTPUT_PATH/temp-file/stage1-sparse-ambient-GS.ply \
    --RATIO 0.1

echo "================================================="
echo "-------------------- STAGE 2 --------------------"
echo "---------- Stage 2.1 - Train Residual Gaussian. ----------"
python train.py \
    $eval \
    --data_device $DATA_DEVICE \
    -s $SOURCE_PATH \
    -m $OUTPUT_PATH \
    -d DA_depth \
    --mesh_path $BASE_MESH_PATH \
    --ply_path $OUTPUT_PATH/temp-file/stage1-sparse-residual-GS.ply \
    --iterations $RESIDUAL_ITERATION

echo "================================================="
echo "---------- Stage 2.2 - BBOX filter. ----------"
python toolkit/bbox_single_scene_xbbox.py \
    --ply_path $OUTPUT_PATH/point_cloud/iteration_$RESIDUAL_ITERATION/point_cloud.ply \
    --blocks_path $BLOCKS_PATH \
    --output $OUTPUT_PATH/temp-file/stage2-bbox-GS.ply

echo "================================================="
echo "---------- Stage 2.3 - Split Residual and Ambient Gaussian. ----------"
python toolkit/split_residual_ambient.py \
    --DATA_DIR $SPLIT_FILE_PATH \
    --PLY_PATH $OUTPUT_PATH/temp-file/stage2-bbox-GS.ply \
    --OUTPUT_PREFIX $OUTPUT_PATH/temp-file/stage2

# ----- Stage 3 -----
echo "================================================="
echo "-------------------- STAGE 3 --------------------"
echo "---------- Stage 3.1 - Combine trained Residual(2.1) and DownSampled Ambient Gaussian(1.4). ----------"
python toolkit/merge_two_ply.py \
    --PLY_PATH_A $OUTPUT_PATH/temp-file/stage1-sparse-ambient-GS.ply \
    --PLY_PATH_B $OUTPUT_PATH/temp-file/stage2-residual.ply \
    --OUTPUT_PLY_PATH $OUTPUT_PATH/temp-file/stage3-start.ply

echo "================================================="
echo "---------- Stage 3.2 - Final Finertune. ----------"
python train.py \
    $eval \
    --data_device $DATA_DEVICE \
    -s $SOURCE_PATH \
    -d DA_depth \
    -m $OUTPUT_PATH \
    --mesh_path $BASE_MESH_PATH \
    --ply_path $OUTPUT_PATH/temp-file/stage3-start.ply \
    --start_iterations $RESIDUAL_ITERATION \
    --iterations $((RESIDUAL_ITERATION + AMBIENT_ITERATION))

echo "================================================="
echo "---------- Stage 3.2 - Render and Metrics. ----------"
python render.py \
    --video \
    --n_frames 300 \
    --skip_train \
    --source_path $TEST_PATH \
    -m $OUTPUT_PATH