CUDA_VISIBLE_DEVICES=0 python3 ../generate_data.py ../../config/hopper.yaml 20 True &
CUDA_VISIBLE_DEVICES=1 python3 ../generate_data.py ../../config/hopper.yaml 21 False &
CUDA_VISIBLE_DEVICES=2 python3 ../generate_data.py ../../config/hopper.yaml 22 False &

wait

CUDA_VISIBLE_DEVICES=0 python3 ../generate_data.py ../../config/hopper.yaml 23 True &
CUDA_VISIBLE_DEVICES=1 python3 ../generate_data.py ../../config/hopper.yaml 24 False &
CUDA_VISIBLE_DEVICES=2 python3 ../generate_data.py ../../config/hopper.yaml 25 False &

wait

CUDA_VISIBLE_DEVICES=0 python3 ../generate_data.py ../../config/hopper.yaml 26 True &
CUDA_VISIBLE_DEVICES=1 python3 ../generate_data.py ../../config/hopper.yaml 27 False &
CUDA_VISIBLE_DEVICES=2 python3 ../generate_data.py ../../config/hopper.yaml 28 False &

wait

CUDA_VISIBLE_DEVICES=0 python3 ../generate_data.py ../../config/hopper.yaml 29 True &
CUDA_VISIBLE_DEVICES=1 python3 ../generate_data.py ../../config/hopper.yaml 30 False &
CUDA_VISIBLE_DEVICES=2 python3 ../generate_data.py ../../config/hopper.yaml 31 False &
