import argparse
import torch

from networks import get_network
from utils.loading import parse_spec
from utils.helpers import get_linear_combination_matrix

DEVICE = "cpu"

class Verifier:
    def __init__(self, net, inputs, eps, true_label):
        self.net = net
        self.inputs = inputs
        self.eps = eps
        self.true_label = true_label

        self.lower_bound = torch.maximum(inputs - eps, torch.zeros_like(inputs))
        self.upper_bound = torch.minimum(inputs + eps, torch.ones_like(inputs))

        self.low_relational = [self.lower_bound.flatten().view(1, -1).clone().T]
        self.up_relational = [self.upper_bound.flatten().view(1, -1).clone().T]
    
    def linear_forward(self, layer):
        weights_positive = torch.maximum(layer.weight, torch.zeros_like(layer.weight))
        weights_negative = torch.minimum(layer.weight, torch.zeros_like(layer.weight))

        lower_bound_new = weights_positive @ self.lower_bound.T + weights_negative @ self.upper_bound.T + layer.bias.view(-1, 1)
        upper_bound_new = weights_positive @ self.upper_bound.T + weights_negative @ self.lower_bound.T + layer.bias.view(-1, 1)
        self.lower_bound = lower_bound_new.T
        self.upper_bound = upper_bound_new.T

        # the relational constraint are same for lower and upper and they are basically the weights and bias (concat) of the layer
        self.low_relational.append(torch.cat((layer.weight, layer.bias.view(-1, 1)), dim=1)) # shape: (out_dim, in_dim + 1) cuz of bias
        self.up_relational.append(torch.cat((layer.weight, layer.bias.view(-1, 1)), dim=1)) # same shape
    
    def conv_forward(self, layer):
        kernel_positive = torch.maximum(layer.weight, torch.zeros_like(layer.weight))
        kernel_negative = torch.minimum(layer.weight, torch.zeros_like(layer.weight))

        input_shape = self.lower_bound.shape

        lower_bound_new = torch.nn.functional.conv2d(self.lower_bound, kernel_positive, bias=layer.bias, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
        upper_bound_new = torch.nn.functional.conv2d(self.upper_bound, kernel_positive, bias=layer.bias, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
        lower_bound_new += torch.nn.functional.conv2d(self.upper_bound, kernel_negative, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
        upper_bound_new += torch.nn.functional.conv2d(self.lower_bound, kernel_negative, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)

        linear_comb_matrix = get_linear_combination_matrix(self.lower_bound, layer)

        self.lower_bound = lower_bound_new
        self.upper_bound = upper_bound_new

        self.low_relational.append(linear_comb_matrix)
        self.up_relational.append(linear_comb_matrix)



    def relu_forward(self, layer):
        pass

    def relu6_forward(self, layer):
        pass
    
    def back_substitute(self):
        curr_lower = self.low_relational[-1] # shape: (L_layer_out_dim, L_layer_in_dim + 1)
        curr_upper = self.up_relational[-1]
        for i in range(len(self.low_relational)-2, -1, -1):
            
            prev_lower = self.low_relational[i] # shape: (L-1_layer_out_dim, L-1_layer_in_dim + 1)
            prev_upper = self.up_relational[i]

            # so curr_lower is basically a constraint of form x_i <= ax_i-1 + bx_i-2 + c (could be many terms) and we want to replace x_i-1 and x_i-2 with even previous constraints. so use similar >0 and <0 trick

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

        self.lower_bound = torch.maximum(self.lower_bound, curr_lower.T)
        self.upper_bound = torch.minimum(self.upper_bound, curr_upper.T)
    
    def check(self):
        res = True
        for i in range(10):
            if i != self.true_label:
                if self.lower_bound[0][self.true_label] <= self.upper_bound[0][i]:
                    res = False
                    break
        return res
    
    def forward(self):
        for layer in self.net:
            print(layer.__class__.__name__)
            if layer.__class__.__name__ == "Linear":
                # flatten
                self.lower_bound = self.lower_bound.flatten().view(1, -1)
                self.upper_bound = self.upper_bound.flatten().view(1, -1)
                self.linear_forward(layer)
            
            elif layer.__class__.__name__ == "Conv2d":
                self.conv_forward(layer)
            
            elif layer.__class__.__name__ == "ReLU":
                self.relu_forward(layer)
            
            elif layer.__class__.__name__ == "ReLU6":
                self.relu6_forward(layer)



def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    """
    Analyzes the given network with the given input and epsilon.
    :param net: Network to analyze.
    :param inputs: Input to analyze.
    :param eps: Epsilon to analyze.
    :param true_label: True label of the input.
    :return: True if the network is verified, False otherwise.
    """
    verifier = Verifier(net, inputs, eps, true_label)
    verifier.forward()
    verifier.back_substitute()
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
    parser.add_argument("--test", type=int, required=False, help="simple: 0 or complex: 1")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    if dataset == "mnist":
        in_ch, in_dim, num_class = 1, 28, 10
    elif dataset == "cifar10":
        in_ch, in_dim, num_class = 3, 32, 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if args.test == 0: # simple test for sanity checks
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 3),
            torch.nn.Linear(3, 1)
        )
        # init net
        net[0].weight = torch.nn.Parameter(torch.tensor([[2.0, -1.0], [1.0, -3.0], [1.0, 2.0]]))
        net[0].bias = torch.nn.Parameter(torch.tensor([1.0, 3.0, 5.0]))
        net[1].weight = torch.nn.Parameter(torch.tensor([[1.0, -1.0, 2.0]]))
        net[1].bias = torch.nn.Parameter(torch.tensor([3.0]))
        
        image = torch.zeros(1,2)
        out = net(image)
        analyze(net, image, 1, 0)
        print(net)

    else:
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

        if analyze(net, image, eps, true_label):
            print("verified")
        else:
            print("not verified")


if __name__ == "__main__":
    main()
