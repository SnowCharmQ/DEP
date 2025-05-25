python -u model-select.py --version 0 --select infer
python -u model-select.py --version 1 --select infer
python -u model-select.py --version 2 --select infer
python -u model-select.py --version 3 --select infer
python -u model-select.py --version 4 --select infer
python -u model-select.py --select eval

python -u model-eval.py --category Books --eval infer --temperature 0.8 --dataset test
python -u model-eval.py --category Books --eval eval --temperature 0.8 --dataset test

python -u model-eval.py --category Movies_and_TV --eval infer --temperature 0.8 --dataset test
python -u model-eval.py --category Movies_and_TV --eval eval --temperature 0.8 --dataset test

python -u model-eval.py --category CDs_and_Vinyl --eval infer --temperature 0.8 --dataset test
python -u model-eval.py --category CDs_and_Vinyl --eval eval --temperature 0.8 --dataset test
