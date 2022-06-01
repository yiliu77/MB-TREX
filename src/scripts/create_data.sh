CUDA_VISIBLE_DEVICES=0 python3 ../generate_data.py ../../config/hopper.yaml 10 &
CUDA_VISIBLE_DEVICES=1 python3 ../generate_data.py ../../config/hopper.yaml 11 &
CUDA_VISIBLE_DEVICES=2 python3 ../generate_data.py ../../config/hopper.yaml 12 &

wait

CUDA_VISIBLE_DEVICES=0 python3 ../generate_data.py ../../config/hopper.yaml 13 &
CUDA_VISIBLE_DEVICES=1 python3 ../generate_data.py ../../config/hopper.yaml 14 &
CUDA_VISIBLE_DEVICES=2 python3 ../generate_data.py ../../config/hopper.yaml 15 &

wait

CUDA_VISIBLE_DEVICES=0 python3 ../generate_data.py ../../config/hopper.yaml 16 &
CUDA_VISIBLE_DEVICES=1 python3 ../generate_data.py ../../config/hopper.yaml 17 &
CUDA_VISIBLE_DEVICES=2 python3 ../generate_data.py ../../config/hopper.yaml 18 &

wait

CUDA_VISIBLE_DEVICES=0 python3 ../generate_data.py ../../config/hopper.yaml 19 &
CUDA_VISIBLE_DEVICES=1 python3 ../generate_data.py ../../config/hopper.yaml 20 &
CUDA_VISIBLE_DEVICES=2 python3 ../generate_data.py ../../config/hopper.yaml 21 &