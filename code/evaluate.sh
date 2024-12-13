#!/bin/bash

# Record the start time of the entire script
script_start_time=$(date +%s)

for net in fc_linear fc_base fc_w fc_d fc_dw fc6_base fc6_w fc6_d fc6_dw conv_linear conv_base conv6_base conv_d skip skip_large skip6 skip6_large
do
    echo "Evaluating network ${net}..."
    
    # Record the start time for this network
    net_start_time=$(date +%s)
    
    for spec in `ls preliminary_test_cases/${net}`
    do
        echo "Evaluating spec ${spec}..."
        
        # Record the start time for this spec
        spec_start_time=$(date +%s)
        
        # Run the verifier
        python code/verifier.py --net ${net} --spec preliminary_test_cases/${net}/${spec}
        
        # Record the end time for this spec
        spec_end_time=$(date +%s)
        spec_duration=$((spec_end_time - spec_start_time))
        
        echo "Spec ${spec} took ${spec_duration} seconds."
    done
    
    # Record the end time for this network
    net_end_time=$(date +%s)
    net_duration=$((net_end_time - net_start_time))
    
    echo "Network ${net} took ${net_duration} seconds."
done

# Record the end time of the entire script
script_end_time=$(date +%s)
script_duration=$((script_end_time - script_start_time))

echo "The entire script took ${script_duration} seconds."

