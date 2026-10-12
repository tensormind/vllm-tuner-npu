
# 9b 单机单卡 
# python aisbench_test.py --input_len 16384 --output 1024 --data_num 64  --concurrency 8
# python aisbench_test.py --input_len 16384 --output 1024 --data_num 128 --concurrency 16
# python aisbench_test.py --input_len 16384 --output 1024 --data_num 128 --concurrency 24
# python aisbench_test.py --input_len 16384 --output 1024 --data_num 192 --concurrency 32
# python aisbench_test.py --input_len 16384 --output 1024 --data_num 192 --concurrency 48 

# python aisbench_test.py --input_len 32768 --output 1000 --data_num 64  --concurrency 8
# python aisbench_test.py --input_len 32768 --output 1000 --data_num 96  --concurrency 16
# python aisbench_test.py --input_len 32768 --output 1000 --data_num 96  --concurrency 24
# python aisbench_test.py --input_len 32768 --output 1000 --data_num 128 --concurrency 32
# python aisbench_test.py --input_len 32768 --output 1000 --data_num 128 --concurrency 48

# 9b 单机8卡 
# 16K
# python aisbench_test.py --input_len 16384 --output 1024 --data_num 128 --concurrency 32  
# python aisbench_test.py --input_len 16384 --output 1024 --data_num 256 --concurrency 64  
# python aisbench_test.py --input_len 16384 --output 1024 --data_num 512 --concurrency 128 

# 32K
# python aisbench_test.py --input_len 32768 --output 1024 --data_num 64  --concurrency 16 
# python aisbench_test.py --input_len 32768 --output 1024 --data_num 128 --concurrency 32 
# python aisbench_test.py --input_len 32768 --output 1024 --data_num 256 --concurrency 64 

#27b 单机8卡

# python aisbench_test.py --input_len 16384 --output 1024 --data_num 128 --concurrency 32  --dataset_type normal
# python aisbench_test.py --input_len 16384 --output 1024 --data_num 256 --concurrency 64  --dataset_type normal
# python aisbench_test.py --input_len 16384 --output 1024 --data_num 512 --concurrency 128 --dataset_type normal
# python aisbench_test.py --input_len 16384 --output 1024 --data_num 768 --concurrency 192 --dataset_type normal

python aisbench_test.py --input_len 32768 --output 1024 --data_num 128 --concurrency 32  --dataset_type normal
python aisbench_test.py --input_len 32768 --output 1024 --data_num 256 --concurrency 64  --dataset_type normal
python aisbench_test.py --input_len 32768 --output 1024 --data_num 256 --concurrency 128 --dataset_type normal
python aisbench_test.py --input_len 32768 --output 1024 --data_num 256 --concurrency 192 --dataset_type normal