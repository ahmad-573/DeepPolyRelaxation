import torch
import torch.nn.functional as F

# Input tensor: 1x1x4x4
x = torch.tensor([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]], dtype=torch.float32)

# Extract sliding patches with kernel_size=2, stride=1, padding=0
unfolded_x = F.unfold(x, kernel_size=2, stride=1, padding=0)

print("Original tensor:")
print(x)

print("\nUnfolded tensor shape:", unfolded_x.shape)
print("Unfolded tensor:")
print(unfolded_x)