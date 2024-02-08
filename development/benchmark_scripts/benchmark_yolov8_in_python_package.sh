output_dir=$1
batch_size=$2
inferences=$3

python -m inference_cli.main benchmark python-package-speed -m yolov8n-640 -bi $inferences -bs $batch_size -o $output_dir/yolov8n_640_bs_$batch_size.json
python -m inference_cli.main benchmark python-package-speed -m yolov8n-1280 -bi $inferences -bs $batch_size -o $output_dir/yolov8n_1280_bs_$batch_size.json

python -m inference_cli.main benchmark python-package-speed -m yolov8s-640 -bi $inferences -bs $batch_size -o $output_dir/yolov8s_640_bs_$batch_size.json
python -m inference_cli.main benchmark python-package-speed -m yolov8s-1280 -bi $inferences -bs $batch_size -o $output_dir/yolov8s_1280_bs_$batch_size.json

python -m inference_cli.main benchmark python-package-speed -m yolov8m-640 -bi $inferences -bs $batch_size -o $output_dir/yolov8m_640_bs_$batch_size.json
python -m inference_cli.main benchmark python-package-speed -m yolov8m-1280 -bi $inferences -bs $batch_size -o $output_dir/yolov8m_1280_bs_$batch_size.json

python -m inference_cli.main benchmark python-package-speed -m yolov8l-640 -bi $inferences -bs $batch_size -o $output_dir/yolov8l_640_bs_$batch_size.json
python -m inference_cli.main benchmark python-package-speed -m yolov8l-1280 -bi $inferences -bs $batch_size -o $output_dir/yolov8l_1280_bs_$batch_size.json

python -m inference_cli.main benchmark python-package-speed -m yolov8x-640 -bi $inferences -bs $batch_size -o $output_dir/yolov8x_640_bs_$batch_size.json
python -m inference_cli.main benchmark python-package-speed -m yolov8x-1280 -bi $inferences -bs $batch_size -o $output_dir/yolov8x_1280_bs_$batch_size.json

python -m inference_cli.main benchmark python-package-speed -m yolov8n-seg-640 -bi $inferences -bs $batch_size -o $output_dir/yolov8n_seg_640_bs_$batch_size.json
python -m inference_cli.main benchmark python-package-speed -m yolov8n-seg-1280 -bi $inferences -bs $batch_size -o $output_dir/yolov8n_seg_1280_bs_$batch_size.json

python -m inference_cli.main benchmark python-package-speed -m yolov8s-seg-640 -bi $inferences -bs $batch_size -o $output_dir/yolov8s_seg_640_bs_$batch_size.json
python -m inference_cli.main benchmark python-package-speed -m yolov8s-seg-1280 -bi $inferences -bs $batch_size -o $output_dir/yolov8s_seg_1280_bs_$batch_size.json

python -m inference_cli.main benchmark python-package-speed -m yolov8m-seg-640 -bi $inferences -bs $batch_size -o $output_dir/yolov8m_seg_640_bs_$batch_size.json
python -m inference_cli.main benchmark python-package-speed -m yolov8m-seg-1280 -bi $inferences -bs $batch_size -o $output_dir/yolov8m_seg_1280_bs_$batch_size.json

python -m inference_cli.main benchmark python-package-speed -m yolov8l-seg-640 -bi $inferences -bs $batch_size -o $output_dir/yolov8l_seg_640_bs_$batch_size.json
python -m inference_cli.main benchmark python-package-speed -m yolov8l-seg-1280 -bi $inferences -bs $batch_size -o $output_dir/yolov8l_seg_1280_bs_$batch_size.json

python -m inference_cli.main benchmark python-package-speed -m yolov8x-seg-640 -bi $inferences -bs $batch_size -o $output_dir/yolov8x_seg_640_bs_$batch_size.json
python -m inference_cli.main benchmark python-package-speed -m yolov8x-seg-1280 -bi $inferences -bs $batch_size -o $output_dir/yolov8x_seg_1280_bs_$batch_size.json
