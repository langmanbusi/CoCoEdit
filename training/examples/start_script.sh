source /home/notebook/data/group/wyh/miniconda3/bin/activate
conda activate nft

cd /home/notebook/data/group/wyh/UniWorld-V2
bash examples/train_qwen_image_edit_0.sh $1 $2
echo running...