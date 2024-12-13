#!/bin/bash

# Default test value is 1
test_value=1

# If --test is provided, set test_value to 0
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            test_value=0
            shift
            ;;
    esac
done

# Run python script with the appropriate test value
python code/verifier.py --net fc_linear --spec test_cases/fc_linear/img_mnist_0.082864.txt --test $test_value
