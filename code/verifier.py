import argparse
import torch

from networks import get_network
from utils.loading import parse_spec
from utils.helpers import get_linear_combination_matrix, clamp

DEVICE = "cpu"

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s(%(funcName)s:%(lineno)d) | %(message)s')

class Verifier:
    def __init__(self, net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int, num_class: int):
        self.net = net
        self.inputs = inputs
        self.eps = eps
        self.true_label = true_label
        self.num_class = num_class

        # Make all layers weights and biases non-trainable
        for layer in self.net:
            for param in layer.parameters():
                param.requires_grad = False

        # Add verification layer
        self.verification_layer = self._get_output_layer(num_class, true_label) 

        # To keep track of skip blocks
        self.skip_starts = []
        self.skip_ends = []

        # Initialize trainable alphas and betas
        self.alphas = {} # slope for lower relational bounds
        self.betas = {} # slope for upper relational bounds
        
        # Initialize optimizer parameters
        self.lr = 0.01
        self.steps = 30
        self.alpha_init = 0.5
        self.beta_init = 0.5

        self.lower_bound = torch.maximum(inputs - eps, torch.zeros_like(inputs))
        self.upper_bound = torch.minimum(inputs + eps, torch.ones_like(inputs))

        self.low_relational = [self.lower_bound.flatten().view(1, -1).clone().T]
        self.up_relational = [self.upper_bound.flatten().view(1, -1).clone().T]
    
    @staticmethod
    def _get_output_layer(num_classes: int, true_label: int) -> torch.nn.Linear:
        '''
        Get the output layer that captures the verification task.
        '''
        verification_layer = torch.nn.Linear(num_classes, num_classes - 1)
        verification_layer.bias = torch.nn.Parameter(torch.zeros(num_classes - 1))
        torch.nn.init.zeros_(verification_layer.weight)

        # The i^th row corresponds to y_true - y_i for i < true_label
        # and y_true - y_{i+1} for i >= true_label
        with torch.no_grad():
            verification_layer.weight[:, true_label] = 1
            verification_layer.weight[:true_label, :true_label].fill_diagonal_(-1)
            verification_layer.weight[true_label:, true_label + 1:].fill_diagonal_(-1)

        # Make sure this layer isn't trained
        for param in verification_layer.parameters():
            param.requires_grad = False

        return verification_layer

    
    def _linear_forward(self, layer: torch.nn.Linear):
        weights_positive = torch.maximum(layer.weight, torch.zeros_like(layer.weight))
        weights_negative = torch.minimum(layer.weight, torch.zeros_like(layer.weight))

        lower_bound_new = weights_positive @ self.lower_bound.T + weights_negative @ self.upper_bound.T + layer.bias.view(-1, 1)
        upper_bound_new = weights_positive @ self.upper_bound.T + weights_negative @ self.lower_bound.T + layer.bias.view(-1, 1)
        self.lower_bound = lower_bound_new.T
        self.upper_bound = upper_bound_new.T

        # the relational constraint are same for lower and upper and they are basically the weights and bias (concat) of the layer
        self.low_relational.append(torch.cat((layer.weight, layer.bias.view(-1, 1)), dim=1)) # shape: (out_dim, in_dim + 1) because of bias
        self.up_relational.append(torch.cat((layer.weight, layer.bias.view(-1, 1)), dim=1)) # same shape
    
    def _conv_forward(self, layer):
        kernel_positive = torch.maximum(layer.weight, torch.zeros_like(layer.weight))
        kernel_negative = torch.minimum(layer.weight, torch.zeros_like(layer.weight))

        lower_bound_new = torch.nn.functional.conv2d(self.lower_bound, kernel_positive, bias=layer.bias, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
        upper_bound_new = torch.nn.functional.conv2d(self.upper_bound, kernel_positive, bias=layer.bias, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
        lower_bound_new += torch.nn.functional.conv2d(self.upper_bound, kernel_negative, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
        upper_bound_new += torch.nn.functional.conv2d(self.lower_bound, kernel_negative, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)

        linear_comb_matrix = get_linear_combination_matrix(self.lower_bound, layer)

        self.lower_bound = lower_bound_new
        self.upper_bound = upper_bound_new

        self.low_relational.append(linear_comb_matrix)
        self.up_relational.append(linear_comb_matrix)


    def _relu_forward(self, layer: torch.nn.ReLU, layer_idx):
        case1_map = (self.upper_bound <= 0).float() # Case 1: upper bound ≤ 0 -> output = 0
        case2_map = (self.lower_bound >= 0).float() # Case 2: lower bound ≥ 0 -> output = input
        case3_map = 1 - case1_map - case2_map        # Case 3: crosses 0 -> need special handling
        
        # Handle Case 1 and Case 2
        lower_bound_new = case1_map * torch.zeros_like(self.lower_bound) + case2_map * self.lower_bound
        upper_bound_new = case1_map * torch.zeros_like(self.upper_bound) + case2_map * self.upper_bound

        low_relational_new = torch.zeros((self.low_relational[-1].shape[0], self.low_relational[-1].shape[0]))
        up_relational_new = low_relational_new.clone()

        low_relational_new += case2_map.flatten().view(1,-1) * torch.eye(self.low_relational[-1].shape[0])
        low_relational_new = torch.cat((low_relational_new, torch.zeros(self.low_relational[-1].shape[0], 1)), dim=1)
        up_relational_new = low_relational_new.clone()

        # Handle Case 3
        lower_bound_new += case3_map * torch.zeros_like(self.lower_bound)  # Lower bound = 0
        upper_bound_new += case3_map * self.upper_bound  # Upper bound = u_i
        
        # Compute slopes (λ) for relational bounds
        slopes = self.upper_bound / (self.upper_bound - self.lower_bound + 1e-8)
        slopes = torch.where(torch.isnan(slopes), torch.zeros_like(slopes), slopes)
        bias = -slopes * self.lower_bound

        slopes = slopes.flatten().view(1, -1)
        bias = bias.flatten().view(1, -1)

        case3_up_relational = torch.cat((slopes * torch.eye(self.low_relational[-1].shape[0]), bias.T), dim=1)
        up_relational_new += case3_map.view(-1, 1) * case3_up_relational

        # Use trainable alpha instead of fixed value
        if layer_idx not in self.alphas:
            self.alphas[layer_idx] = torch.nn.Parameter(
                torch.ones_like(self.lower_bound) * self.alpha_init
            )
        # clamp between 0 and 1 inclusive
        alpha = clamp(self.alphas[layer_idx], 0, 1)
        case3_low_relational = torch.cat((alpha.view(-1, 1) * torch.eye(self.low_relational[-1].shape[0]), 
                                        torch.zeros_like(bias.T)), dim=1)

        low_relational_new += case3_map.view(-1, 1) * case3_low_relational

        self.lower_bound = lower_bound_new
        self.upper_bound = upper_bound_new
        self.low_relational.append(low_relational_new)
        self.up_relational.append(up_relational_new)

    def _relu6_forward(self, layer: torch.nn.ReLU6, layer_idx):
        case1_map = (self.upper_bound <= 0).float()  # same as relu case 1
        case2_map = (self.lower_bound >= 6).float() # same as relu case 1 but with 6
        case3_map = (self.lower_bound >= 0).float() * (self.upper_bound < 6).float() * (self.upper_bound > 0).float() # same as relu case 2
        case4_map = (self.lower_bound < 0).float() * (self.upper_bound <= 6).float() * (self.upper_bound > 0).float() # same as relu case 3 - crossing relu (need alphas) 
        case5_map = (self.upper_bound >= 6).float() * (self.lower_bound < 6).float() * (self.lower_bound >= 0).float() # similar to relu case 3 - crossing relu
        case6_map = (self.lower_bound < 0).float() * (self.upper_bound > 6).float() # new case with two crossings. need alpha and beta

        # assert (case1_map + case2_map + case3_map + case4_map + case5_map + case6_map == 1).all()

        # Case 1
        lower_bound_new = case1_map * torch.zeros_like(self.lower_bound) # Lower bound = 0
        upper_bound_new = case1_map * torch.zeros_like(self.upper_bound) # Upper bound = 0

        low_relational_new = torch.zeros((self.low_relational[-1].shape[0], self.low_relational[-1].shape[0]))
        up_relational_new = low_relational_new.clone()

        # Case 2
        lower_bound_new += case2_map * torch.ones_like(self.lower_bound) * 6 # Lower bound = 6
        upper_bound_new += case2_map * torch.ones_like(self.upper_bound) * 6 # Upper bound = 6

        # Keep track of bias terms for relational constraints
        low_append = case2_map.flatten().view(-1,1) * 6
        up_append = low_append.clone()

        # Case 3
        lower_bound_new += case3_map * self.lower_bound # Lower bound = l_i
        upper_bound_new += case3_map * self.upper_bound # Upper bound = u_i

        low_relational_new += case3_map.flatten().view(1,-1) * torch.eye(self.low_relational[-1].shape[0])
        up_relational_new = low_relational_new.clone()

        # Case 4
        lower_bound_new += case4_map * torch.zeros_like(self.lower_bound)  # Lower bound = 0
        upper_bound_new += case4_map * self.upper_bound # Upper bound = u_i
        
        # Compute slopes (λ) for relational bounds
        slopes = self.upper_bound / (self.upper_bound - self.lower_bound + 1e-8)
        slopes = torch.where(torch.isnan(slopes), torch.zeros_like(slopes), slopes)
        bias = -slopes * self.lower_bound

        slopes = slopes.flatten().view(1, -1)
        bias = bias.flatten().view(1, -1)

        case4_up_relational = slopes * torch.eye(self.low_relational[-1].shape[0])
        up_append += case4_map.flatten().view(-1,1) * bias.T

        up_relational_new += case4_map.view(-1, 1) * case4_up_relational

        if layer_idx not in self.alphas:
            self.alphas[layer_idx] = torch.nn.Parameter(
                torch.ones_like(self.lower_bound) * self.alpha_init
            )
        alpha = clamp(self.alphas[layer_idx], 0, 1)
        case4_low_relational = alpha.view(-1, 1) * torch.eye(self.low_relational[-1].shape[0])
        
        low_relational_new += case4_map.view(-1, 1) * case4_low_relational

        #Case 5
        upper_bound_new += case5_map * torch.ones_like(self.lower_bound) * 6 # Upper bound = 6
        lower_bound_new += case5_map * self.lower_bound # Lower bound = l_i

        # Compute slopes (λ) for relational bounds
        slopes = (torch.ones_like(self.lower_bound) * 6 - self.lower_bound) / (self.upper_bound - self.lower_bound + 1e-8)
        slopes = torch.where(torch.isnan(slopes), torch.zeros_like(slopes), slopes)
        bias = (torch.ones_like(slopes) - slopes) * self.lower_bound

        slopes = slopes.flatten().view(1, -1)
        bias = bias.flatten().view(1, -1)

        case5_low_relational = slopes * torch.eye(self.low_relational[-1].shape[0])
        low_append += case5_map.flatten().view(-1,1) * bias.T
        low_relational_new += case5_map.view(-1, 1) * case5_low_relational

        if layer_idx not in self.betas:
            self.betas[layer_idx] = torch.nn.Parameter(
                torch.ones_like(self.upper_bound) * self.beta_init
            )
        beta = clamp(self.betas[layer_idx], 0, 1)
        case5_up_relational = beta.view(-1, 1) * torch.eye(self.low_relational[-1].shape[0])
        up_append += case5_map.flatten().view(-1,1) * ((torch.ones_like(bias.T) * 6) - (beta.view(-1, 1) * 6))

        up_relational_new += case5_map.view(-1, 1) * case5_up_relational

        # Case 6
        lower_bound_new += case6_map * torch.zeros_like(self.lower_bound)  # Lower bound = 0
        upper_bound_new += case6_map * torch.ones_like(self.upper_bound) * 6  # Upper bound = 6

        # calculate the lower relational constraint
        # y >= (6 \alpha / ux) x 
        if layer_idx not in self.alphas:
            self.alphas[layer_idx] = torch.nn.Parameter(
                torch.ones_like(self.lower_bound) * self.alpha_init
            )
        alpha = clamp(self.alphas[layer_idx], 0, 1)
        slope = (6 * alpha) / (self.upper_bound + 1e-8)
        case6_low_relational = slope.view(-1, 1) * torch.eye(self.low_relational[-1].shape[0])

        low_relational_new += case6_map.view(-1, 1) * case6_low_relational

        # calculate the upper relational constraint
        # y <= (6 \alpha / (6 - lx)) x + (6 - 36 \alpha / (6 - lx))
        if layer_idx not in self.betas:
            self.betas[layer_idx] = torch.nn.Parameter(
                torch.ones_like(self.upper_bound) * self.beta_init
            )
        beta = clamp(self.betas[layer_idx], 0, 1)
        slope = (6 * beta) / (6 - self.lower_bound + 1e-8)
        bias = 6 - (36 * beta) / (6 - self.lower_bound + 1e-8)

        case6_up_relational = slope.view(-1, 1) * torch.eye(self.low_relational[-1].shape[0])
        up_append += case6_map.view(-1,1) * bias.view(-1, 1)

        up_relational_new += case6_map.view(-1, 1) * case6_up_relational

        low_relational_new = torch.cat((low_relational_new, low_append), dim=1)
        up_relational_new = torch.cat((up_relational_new, up_append), dim=1)

        self.lower_bound = lower_bound_new
        self.upper_bound = upper_bound_new
        self.low_relational.append(low_relational_new)
        self.up_relational.append(up_relational_new)
                                        

    def _skip_connection_forward(self, layer, layer_idx):
        curr_lower_bound = self.lower_bound.clone()
        curr_upper_bound = self.upper_bound.clone()

        self.skip_starts.append(len(self.low_relational))
        
        for i, sub_layer in enumerate(layer.path):
            if sub_layer.__class__.__name__ == "Linear":
                self.lower_bound = self.lower_bound.flatten().view(1, -1)
                self.upper_bound = self.upper_bound.flatten().view(1, -1)
                self._linear_forward(sub_layer)
                if i == len(layer.path) - 1:
                    self.skip_ends.append(len(self.low_relational))
                self._back_substitute()
            
            if sub_layer.__class__.__name__ == "ReLU":
                self._relu_forward(sub_layer, (layer_idx, i))
            
            elif sub_layer.__class__.__name__ == "ReLU6":
                self._relu6_forward(sub_layer, (layer_idx, i))

        self.lower_bound = curr_lower_bound + self.lower_bound  
        self.upper_bound = curr_upper_bound + self.upper_bound
        

    
    def _back_substitute(self):
        curr_lower = self.low_relational[-1] # shape: (L_layer_out_dim, L_layer_in_dim + 1)
        curr_upper = self.up_relational[-1]

        add_low = 0
        add_up = 0

        starts = self.skip_starts.copy() # Layer numbers where skip connections start 
        ends = self.skip_ends.copy() # Layer numbers where skip connections end

        if len(self.low_relational) - 1  in ends:
            add_low = curr_lower.clone()
            add_up = curr_upper.clone()
            # make last column 0
            add_low = torch.cat((add_low[:, :-1], torch.zeros(add_low.shape[0], 1)), dim=1)
            add_up = torch.cat((add_up[:, :-1], torch.zeros(add_up.shape[0], 1)), dim=1)
            
        i = len(self.low_relational) - 2
        while i > -1:
            if len(starts) != len(ends):
                # remove last element of starts
                starts.pop() 
                # assert len(starts) == len(ends)
                continue
            prev_lower = self.low_relational[i] # shape: (L-1_layer_out_dim, L-1_layer_in_dim + 1)
            prev_upper = self.up_relational[i]

            # curr_lower is a constraint of the form x_i <= ax_i-1 + bx_i-2 + c (could be many terms) and we want to replace x_i-1 and x_i-2 with even previous constraints. so we use similar >0 and <0 trick

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

            if i in starts:
                curr_lower += add_low
                curr_upper += add_up

            if i in ends:
                add_low = curr_lower.clone()
                add_up = curr_upper.clone()
                # make last column 0
                add_low = torch.cat((add_low[:, :-1], torch.zeros(add_low.shape[0], 1)), dim=1)
                add_up = torch.cat((add_up[:, :-1], torch.zeros(add_up.shape[0], 1)), dim=1)
            
            i -= 1


        shape = self.lower_bound.shape

        self.lower_bound = torch.maximum(self.lower_bound.flatten().view(1,-1), curr_lower.T)
        self.upper_bound = torch.minimum(self.upper_bound.flatten().view(1,-1), curr_upper.T)

        self.lower_bound = self.lower_bound.view(shape)
        self.upper_bound = self.upper_bound.view(shape)
    
    def check(self):
        # Since the verification layer computes y_true - y_other,
        # we just need to check if all lower bounds are positive
        return (self.lower_bound > 0).all()
    
    def forward(self):
        # Reset bounds, relational constraints, and skip indices at the start of each forward pass
        self.skip_starts = []
        self.skip_ends = []
        self.lower_bound = torch.maximum(self.inputs - self.eps, torch.zeros_like(self.inputs))
        self.upper_bound = torch.minimum(self.inputs + self.eps, torch.ones_like(self.inputs))
        
        self.low_relational = [self.lower_bound.flatten().view(1, -1).clone().T]
        self.up_relational = [self.upper_bound.flatten().view(1, -1).clone().T]

        for i, layer in enumerate(self.net):
            # check if lower bound < upper bound
            # assert (self.lower_bound <= self.upper_bound).all()
            if layer.__class__.__name__ == "Linear":
                self.lower_bound = self.lower_bound.flatten().view(1, -1)
                self.upper_bound = self.upper_bound.flatten().view(1, -1)
                self._linear_forward(layer)
                self._back_substitute()
            
            elif layer.__class__.__name__ == "Conv2d":
                self._conv_forward(layer)
                self._back_substitute()
            
            elif layer.__class__.__name__ == "ReLU":
                self._relu_forward(layer, i)
            
            elif layer.__class__.__name__ == "ReLU6":
                self._relu6_forward(layer, i)
            
            elif layer.__class__.__name__ == "SkipBlock":
                self._skip_connection_forward(layer, i)

        # Apply verification layer at the end
        self._linear_forward(self.verification_layer)
        self._back_substitute()

    def optimize(self):
        """Optimize alpha and beta parameters to improve verification precision."""
        # Do one forward pass to create the alpha parameters
        self.forward()
        
        # Check if we already succeeded
        if self.check():
            return True
            
        # Return false if no trainable parameters
        if (len(self.alphas) + len(self.betas)) == 0:
            return False
        
        # Now create optimizer with the created parameters - alphas and betas
        optimizer = torch.optim.RMSprop(list(self.alphas.values()) + list(self.betas.values()), lr=self.lr, alpha=0.999)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)


        for step in range(self.steps):
            optimizer.zero_grad()
            
            self.forward()
            
            outputs = self.lower_bound
            min_output = outputs.min()
            
            # If verified, we can stop
            if min_output > 0:
                return True
                
            # Loss is negative of minimum output (we want to maximize it)
            loss = -min_output
                
            loss.backward()
            
            optimizer.step()
            scheduler.step()
        
        return False

def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int, num_class: int
) -> bool:
    """
    Analyzes the given network with the given input and epsilon.
    :param net: Network to analyze.
    :param inputs: Input to analyze.
    :param eps: Epsilon to analyze.
    :param true_label: True label of the input.
    :return: True if the network is verified, False otherwise.
    """
    verifier = Verifier(net, inputs, eps, true_label, num_class)
    
    # Try to verify with optimization
    if verifier.optimize():
        return True
    
    # If optimization fails, do one final forward pass and check
    verifier.forward()
    return verifier.check()

def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_linear",
            "fc_base",
            "fc_w",
            "fc_d",
            "fc_dw",
            "fc6_base",
            "fc6_w",
            "fc6_d",
            "fc6_dw",
            "conv_linear",
            "conv_base",
            "conv6_base",
            "conv_d",
            "skip",
            "skip_large",
            "skip6",
            "skip6_large",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    if dataset == "mnist":
        in_ch, in_dim, num_class = 1, 28, 10
    elif dataset == "cifar10":
        in_ch, in_dim, num_class = 3, 32, 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    net = get_network(
        args.net,
        in_ch=in_ch,
        in_dim=in_dim,
        num_class=num_class,
        weight_path=f"models/{dataset}_{args.net}.pt",
    ).to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label, num_class):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
