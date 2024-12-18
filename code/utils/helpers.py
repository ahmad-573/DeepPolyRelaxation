import torch
import torch.nn.functional as F
import torch.nn as nn

def generate_matrix_sequence_with_conv(n, m, K, stride_x, stride_y):
    """
    Generate a sequence of matrices of size (n x m) with a sub-matrix K moving across the matrix using transposed convolution.
    
    Parameters:
        n (int): Number of rows in the matrix.
        m (int): Number of columns in the matrix.
        K (torch.Tensor): Sub-matrix of size (a x b).
        stride_x (int): Horizontal stride for moving K.
        stride_y (int): Vertical stride for moving K.
    
    Returns:
        torch.Tensor: A tensor of shape (num_matrices, n, m) where each slice is a matrix.
    """
    # Dimensions of the sub-matrix K
    a, b = K.shape
    
    # Calculate the output size (number of matrices in the sequence)
    out_h = (n - a) // stride_y + 1
    out_w = (m - b) // stride_x + 1
    num_matrices = out_h * out_w
    
    # Create a one-hot encoded tensor for the positions of K (convert to float)
    position_matrix = torch.eye(num_matrices).view(num_matrices, 1, out_h, out_w).float()
    
    # Calculate output padding to match the desired matrix size
    output_padding_y = (n - (out_h - 1) * stride_y - a) if stride_y > 1 else 0
    output_padding_x = (m - (out_w - 1) * stride_x - b) if stride_x > 1 else 0
    
    # Perform transposed convolution to scatter K (convert K to float)
    output = F.conv_transpose2d(
        position_matrix,              # Input: position indicators
        K.float().unsqueeze(0).unsqueeze(0), # Kernel: K as a 4D tensor (1, 1, a, b)
        stride=(stride_y, stride_x), # Stride for placing K
        output_padding=(output_padding_y, output_padding_x),
    )
    
    # Return the generated sequence of matrices
    return output.squeeze(1)

def get_linear_combination_matrix(input_tensor, conv_layer):
    # Input dimensions
    C, H, W = input_tensor.shape  # Input has C channels, height H, width W
    
    # Convolution parameters
    D = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    
    # Step 1: Unfold the input
    unfolded_x = F.unfold(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)  # Shape: (C * kernel_size^2, H' * W')
    H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

    # Step 2: Reshape the kernel
    conv_weight_flat = conv_layer.weight.view(D, -1)  # Shape: (D, C * kernel_size^2)

    # Step 3: Initialize the matrix for linear combination coefficients
    num_inputs = C * H * W + 1
    num_outputs = D * H_out * W_out
    linear_comb_matrix = torch.zeros(num_outputs, num_inputs, dtype=input_tensor.dtype)

    # Flatten the input tensor
    input_flat = input_tensor.view(-1)

    # Step 4: Populate the linear combination matrix
    for j in range(num_outputs):
        patch_idx = j % (H_out * W_out)  # Patch index in unfolded_x
        channel_idx = j // (H_out * W_out)  # Output channel
        weights_for_yj = conv_weight_flat[channel_idx]  # Weights corresponding to the output channel

        # Extract the patch from unfolded_x
        x_indices = unfolded_x[:, patch_idx]  # Shape: (C * kernel_size^2,)

        # Find the corresponding x_i indices
        for k in range(x_indices.size(0)):
            input_idx = k // (kernel_size[0] * kernel_size[1])
            row_offset = (k % (kernel_size[0] * kernel_size[1])) // kernel_size[1]
            col_offset = k % kernel_size[1]
            
            row = (patch_idx // W_out) * stride[0] - padding[0] + row_offset * dilation[0]
            col = (patch_idx % W_out) * stride[1] - padding[1] + col_offset * dilation[1]
            
            if 0 <= row < H and 0 <= col < W:
                flat_idx = input_idx * H * W + row * W + col
                linear_comb_matrix[j, flat_idx] = weights_for_yj[k]
        
        # append bias
        linear_comb_matrix[j, -1] = conv_layer.bias[channel_idx]

    return linear_comb_matrix

def clamp(x, min_val, max_val):
    return torch.clamp(x, min_val, max_val)

def pseudo_back_sub(low_list, up_list):
    curr_lower = low_list[-1] # shape: (L_layer_out_dim, L_layer_in_dim + 1)
    curr_upper = up_list[-1]

    for i in range(len(low_list) - 2, -1, -1):
        prev_lower = low_list[i]
        prev_upper = up_list[i]

        lower_positive = torch.maximum(curr_lower, torch.zeros_like(curr_lower)) # shape: (L_layer_out_dim, L_layer_in_dim + 1)
        lower_negative = torch.minimum(curr_lower, torch.zeros_like(curr_lower)) # shape: (L_layer_out_dim, L_layer_in_dim + 1)

        upper_positive = torch.maximum(curr_upper, torch.zeros_like(curr_upper))
        upper_negative = torch.minimum(curr_upper, torch.zeros_like(curr_upper))

        # append a 0s row with last entry 1 to the prev_lower and prev_upper
        append_this = torch.zeros(1, prev_lower.shape[1])
        append_this[0][-1] = 1
        prev_lower_append = torch.cat((prev_lower, append_this), dim=0)
        prev_upper_append = torch.cat((prev_upper, append_this), dim=0)

        curr_lower = lower_positive @ prev_lower_append + lower_negative @ prev_upper_append
        curr_upper = upper_positive @ prev_upper_append + upper_negative @ prev_lower_append
    
    return curr_lower, curr_upper

if __name__ == "__main__":
    # x = torch.tensor([[[1, 2, 3],
    #                 [4, 5, 6],
    #                 [7, 8, 9]]], dtype=torch.float32)
    # # Convolution: Single 2x2 kernel, no bias
    # conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, bias=False)
    # conv_layer.weight.data = torch.tensor([[[[2, 3],
    #                                         [-1, 4]]]], dtype=torch.float32)  # Kernel: [[1, 0], [0, -1]]

    x = torch.randn(1, 28, 28)
    # append a 1s channel
    conv_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2, bias=True)
    # random kernel
    conv_layer.weight.data = torch.randn(16, 1, 3, 3)

    # print(conv_layer.bias.shape)

    # Compute the linear combination matrix
    linear_comb_matrix = get_linear_combination_matrix(x, conv_layer)

    # Verify output
    # print("Input tensor (flattened):")
    # print(x.view(-1))

    # print("\nConvolution kernel:")
    # print(conv_layer.weight)

    # print("\nLinear combination matrix:")
    # print(linear_comb_matrix)

    # Compute convolution manually and verify
    output = conv_layer(x)
    print("\nOutput from convolution:")
    print(output)

    # # Reconstruct output using the linear combination matrix
    x = x.view(-1)
    # append a 1s column to x
    x = torch.cat([x, torch.ones(1, dtype=x.dtype)])
    print(x.shape)
    
    reconstructed_output = linear_comb_matrix @ x
    print("\nReconstructed output:")
    print(reconstructed_output.view_as(output))

    assert torch.allclose(output, reconstructed_output.view_as(output), atol=1e-6), "Output mismatch"
