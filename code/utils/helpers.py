import torch
import torch.nn.functional as F
import torch.nn as nn

def get_linear_combination_matrix(input_tensor, conv_layer):
    '''
    Represent the convolution operation as an affine operation.
    '''
    C, H, W = input_tensor.shape  # Input has C channels, height H, width W
    
    # Convolution parameters
    D = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    
    # Unfold input
    unfolded_x = F.unfold(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)  # Shape: (C * kernel_size^2, H' * W')
    H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

    # Reshape the kernel
    conv_weight_flat = conv_layer.weight.view(D, -1)  # Shape: (D, C * kernel_size^2)

    # Initialize the matrix for linear combination coefficients
    num_inputs = C * H * W + 1
    num_outputs = D * H_out * W_out
    linear_comb_matrix = torch.zeros(num_outputs, num_inputs, dtype=input_tensor.dtype)

    input_flat = input_tensor.view(-1)

    # Step 4: Populate the linear combination matrix
    for j in range(num_outputs):
        patch_idx = j % (H_out * W_out)  # Patch index in unfolded_x
        channel_idx = j // (H_out * W_out)  # Output channel
        weights_for_yj = conv_weight_flat[channel_idx]  # Weights corresponding to the output channel

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
