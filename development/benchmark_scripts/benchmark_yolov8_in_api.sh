output_dir=$1
batch_size=$2
clients=$3
requests=$4


python -m inference_cli.main benchmark api-speed -m yolov8n-640 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8n_640_bs_${batch_size}_clients_${clients}_via_http.json
python -m inference_cli.main benchmark api-speed -m yolov8n-1280 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8n_1280_bs_${batch_size}_clients_${clients}_via_http.json

python -m inference_cli.main benchmark api-speed -m yolov8s-640 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8s_640_bs_${batch_size}_clients_${clients}_via_http.json
python -m inference_cli.main benchmark api-speed -m yolov8s-1280 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8s_1280_bs_${batch_size}_clients_${clients}_via_http.json

python -m inference_cli.main benchmark api-speed -m yolov8m-640 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8m_640_bs_${batch_size}_clients_${clients}_via_http.json
python -m inference_cli.main benchmark api-speed -m yolov8m-1280 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8m_1280_bs_${batch_size}_clients_${clients}_via_http.json

python -m inference_cli.main benchmark api-speed -m yolov8l-640 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8l_640_bs_${batch_size}_clients_${clients}_via_http.json
python -m inference_cli.main benchmark api-speed -m yolov8l-1280 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8l_1280_bs_${batch_size}_clients_${clients}_via_http.json

python -m inference_cli.main benchmark api-speed -m yolov8x-640 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8x_640_bs_${batch_size}_clients_${clients}_via_http.json
python -m inference_cli.main benchmark api-speed -m yolov8x-1280 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8x_1280_bs_${batch_size}_clients_${clients}_via_http.json

python -m inference_cli.main benchmark api-speed -m yolov8n-seg-640 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8n_seg_640_bs_${batch_size}_clients_${clients}_via_http.json
python -m inference_cli.main benchmark api-speed -m yolov8n-seg-1280 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8n_seg_1280_bs_${batch_size}_clients_${clients}_via_http.json

python -m inference_cli.main benchmark api-speed -m yolov8s-seg-640 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8s_seg_640_bs_${batch_size}_clients_${clients}_via_http.json
python -m inference_cli.main benchmark api-speed -m yolov8s-seg-1280 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8s_seg_1280_bs_${batch_size}_clients_${clients}_via_http.json

python -m inference_cli.main benchmark api-speed -m yolov8m-seg-640 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8m_seg_640_bs_${batch_size}_clients_${clients}_via_http.json
python -m inference_cli.main benchmark api-speed -m yolov8m-seg-1280 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8m_seg_1280_bs_${batch_size}_clients_${clients}_via_http.json

python -m inference_cli.main benchmark api-speed -m yolov8l-seg-640 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8l_seg_640_bs_${batch_size}_clients_${clients}_via_http.json
python -m inference_cli.main benchmark api-speed -m yolov8l-seg-1280 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8l_seg_1280_bs_${batch_size}_clients_${clients}_via_http.json

python -m inference_cli.main benchmark api-speed -m yolov8x-seg-640 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8x_seg_640_bs_${batch_size}_clients_${clients}_via_http.json
python -m inference_cli.main benchmark api-speed -m yolov8x-seg-1280 -c $clients -br $requests -bs $batch_size -o ${output_dir}/yolov8x_seg_1280_bs_${batch_size}_clients_${clients}_via_http.json
