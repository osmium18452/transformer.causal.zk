torchrun --nproc_per_node=4 --nnodes=1  main.py -MfGN std -C 4,5,6,7 -e 100 -b 1024 -i 48 -p 24
torchrun --nproc_per_node=4 --nnodes=1  main.py -MfGN std -C 4,5,6,7 -e 100 -b 512 -i 48 -p 48
torchrun --nproc_per_node=4 --nnodes=1  main.py -MfGN std -C 4,5,6,7 -e 100 -b 256 -i 48 -p 96
torchrun --nproc_per_node=4 --nnodes=1  main.py -MfGN std -C 4,5,6,7 -e 100 -b 128 -i 48 -p 168
torchrun --nproc_per_node=4 --nnodes=1  main.py -MfGN std -C 4,5,6,7 -e 100 -b 64 -i 48 -p 200
torchrun --nproc_per_node=4 --nnodes=1  main.py -MfGN std -C 4,5,6,7 -e 100 -b 32 -i 48 -p 336

#torchrun --nproc_per_node=4 --nnodes=1  main.py -MfGN std -C 4,5,6,7 -e 2 -b 1024 -i 48 -p 24
#torchrun --nproc_per_node=4 --nnodes=1  main.py -MfGN std -C 4,5,6,7 -e 2 -b 512 -i 48 -p 48
#torchrun --nproc_per_node=4 --nnodes=1  main.py -MfGN std -C 4,5,6,7 -e 2 -b 256 -i 48 -p 96
#torchrun --nproc_per_node=4 --nnodes=1  main.py -MfGN std -C 4,5,6,7 -e 2 -b 128 -i 48 -p 168
#torchrun --nproc_per_node=4 --nnodes=1  main.py -MfGN std -C 4,5,6,7 -e 2 -b 64 -i 48 -p 200
#torchrun --nproc_per_node=4 --nnodes=1  main.py -MfGN std -C 4,5,6,7 -e 2 -b 32 -i 48 -p 336