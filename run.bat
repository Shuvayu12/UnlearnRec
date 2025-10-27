@echo off
REM UnlearnRec - Quick Start Scripts for Windows

echo ========================================
echo UnlearnRec Quick Start Menu
echo ========================================
echo.
echo Select an option:
echo 1. Train base model on MovieLens-1M (quick test - 10 epochs)
echo 2. Train base model on MovieLens-1M (full - 100 epochs)
echo 3. Train and unlearn on MovieLens-1M (quick test)
echo 4. Train and unlearn on MovieLens-1M (full)
echo 5. Train on Gowalla dataset
echo 6. Train on Yelp2018 dataset
echo 7. Custom command (manual)
echo 8. Exit
echo.

set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" (
    echo Running quick base model training...
    python train_and_evaluate.py --mode train --dataset movielens-1m --model lightgcn --num_pretrain_epochs 10 --eval_k 10 --save_model
    goto end
)

if "%choice%"=="2" (
    echo Running full base model training...
    python train_and_evaluate.py --mode train --dataset movielens-1m --model lightgcn --num_pretrain_epochs 100 --eval_k 20 --save_model
    goto end
)

if "%choice%"=="3" (
    echo Running quick training and unlearning...
    python train_and_evaluate.py --mode both --dataset movielens-1m --model lightgcn --num_pretrain_epochs 10 --eval_k 10 --unlearn_test_ratio 0.05 --fine_tune --save_model
    goto end
)

if "%choice%"=="4" (
    echo Running full training and unlearning...
    python train_and_evaluate.py --mode both --dataset movielens-1m --model lightgcn --num_pretrain_epochs 100 --eval_k 20 --unlearn_test_ratio 0.1 --fine_tune --fine_tune_epochs 5 --save_model
    goto end
)

if "%choice%"=="5" (
    echo Training on Gowalla dataset...
    python train_and_evaluate.py --mode both --dataset gowalla --model lightgcn --num_pretrain_epochs 100 --eval_k 20 --save_model
    goto end
)

if "%choice%"=="6" (
    echo Training on Yelp2018 dataset...
    python train_and_evaluate.py --mode both --dataset yelp2018 --model sgl --num_pretrain_epochs 100 --eval_k 20 --save_model
    goto end
)

if "%choice%"=="7" (
    echo.
    echo Enter your custom command (without 'python train_and_evaluate.py'):
    set /p custom_cmd="Arguments: "
    python train_and_evaluate.py %custom_cmd%
    goto end
)

if "%choice%"=="8" (
    echo Exiting...
    exit /b 0
)

echo Invalid choice. Please run the script again.

:end
echo.
echo ========================================
echo Operation completed!
echo ========================================
echo.
echo Checkpoints saved in: checkpoints\
echo Results saved in: results\
echo.
pause
