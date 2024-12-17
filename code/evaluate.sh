#!/bin/bash

# Record the start time of the entire script
script_start_time=$(date +%s)

total_correct=0
total_output=0

incorrect_cases=()

for net in fc_linear fc_base fc_w fc_d fc_dw fc6_base fc6_w fc6_d fc6_dw conv_linear conv_base conv6_base conv_d skip skip_large skip6 skip6_large
do
    echo "Evaluating network ${net}..."
    
    # Record the start time for this network
    net_start_time=$(date +%s)
    
    for spec in `ls test_cases/${net}`
    do
        echo "Evaluating spec ${spec}..."
        
        # Record the start time for this spec
        spec_start_time=$(date +%s)

        # Run the verifier
        output=$(python code/verifier.py --net ${net} --spec test_cases/${net}/${spec})
        # Check the output and update counters
        if echo "$output" | grep -q " correct"; then
            total_correct=$((total_correct + 1))
			total_output=$((total_output + 1))
        fi

		if echo "$output" | grep -q "incorrect"; then
            incorrect_cases+=("${net}/${spec}-${output}")
			total_output=$((total_output + 1))
        fi
    
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

# Output the final summary
echo -e "\nTotal Corrects: ${total_correct}"
echo -e "\nTotal Outputs: ${total_output}"
echo -e "\nThe entire script took ${script_duration} seconds.\n"

if [ ${#incorrect_cases[@]} -gt 0 ]; then
	echo "Incorrect cases:"
	for case in "${incorrect_cases[@]}"
	do
		echo "${case}"
	done
fi
