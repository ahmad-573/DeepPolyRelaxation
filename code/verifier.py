import argparse
import torch

from networks import get_network
from utils.loading import parse_spec

DEVICE = "cpu"

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
    res = True
    # go through all the layers
    lower_bound_first = torch.maximum(inputs - eps, torch.zeros_like(inputs))
    upper_bound_first = torch.minimum(inputs + eps, torch.ones_like(inputs))
    # lower_bound_first = inputs - eps
    # upper_bound_first = inputs + eps
    # lower_bound = torch.zeros(1, 2) - 1
    # upper_bound = torch.zeros(1, 2) + 1

    lower_bound = lower_bound_first.flatten().view(1, -1)
    upper_bound = upper_bound_first.flatten().view(1, -1)

    low_relational = [lower_bound.clone().T]
    up_relational = [upper_bound.clone().T]

    # print(lower_bound.shape, upper_bound.shape)
    for layer in net:
        print(layer.__class__.__name__)
        if layer.__class__.__name__ == "Linear":
            weights_positive = torch.maximum(layer.weight, torch.zeros_like(layer.weight))
            weights_negative = torch.minimum(layer.weight, torch.zeros_like(layer.weight))

            lower_bound_new = weights_positive @ lower_bound.T + weights_negative @ upper_bound.T + layer.bias.view(-1, 1)
            upper_bound_new = weights_positive @ upper_bound.T + weights_negative @ lower_bound.T + layer.bias.view(-1, 1)

            lower_bound = lower_bound_new.T
            upper_bound = upper_bound_new.T

            # the relational constraint are same for lower and upper and they are basically the weights and bias (concat) of the layer
            low_relational.append(torch.cat((layer.weight, layer.bias.view(-1, 1)), dim=1)) # shape: (out_dim, in_dim + 1) cuz of bias
            up_relational.append(torch.cat((layer.weight, layer.bias.view(-1, 1)), dim=1)) # same shape
    

    # backsubstitute the bounds (only for linear at this point)
    # coeffs = torch.eye(inputs.shape[-1] * inputs.shape[-2])
    # for i, layer in enumerate(net):
    #     if layer.__class__.__name__ == "Linear":
    #         constant = torch.zeros(layer.weight.shape[1], 1)
    #         break
    
    # for i, layer in enumerate(net):
    #     if layer.__class__.__name__ == "Linear":
    #         coeffs = layer.weight @ coeffs
    #         constant = layer.weight @ constant + layer.bias.view(-1, 1) 
    
    # coeffs_positive = torch.maximum(coeffs, torch.zeros_like(coeffs))
    # coeffs_negative = torch.minimum(coeffs, torch.zeros_like(coeffs))

    # lower_bound_back = coeffs_positive @ lower_bound_first.flatten().view(-1, 1) + coeffs_negative @ upper_bound_first.flatten().view(-1, 1) + constant
    # upper_bound_back = coeffs_positive @ upper_bound_first.flatten().view(-1, 1) + coeffs_negative @ lower_bound_first.flatten().view(-1, 1) + constant

    # # print(lower_bound.shape, upper_bound.shape, lower_bound_back.shape, upper_bound_back.shape)
    
    # # print(lower_bound, upper_bound)

    # lower_bound = torch.maximum(lower_bound, lower_bound_back.T)
    # upper_bound = torch.minimum(upper_bound, upper_bound_back.T)

    # print(lower_bound, upper_bound)

    low_rel, up_rel = back_substitute(low_relational, up_relational)
    # print(low_rel, up_rel)
    # print(lower_bound, upper_bound)
    lower_bound = torch.maximum(lower_bound, low_rel.T)
    upper_bound = torch.minimum(upper_bound, up_rel.T)

    # print(lower_bound, upper_bound)

    # print("HELOO", lower_bound, upper_bound)

    # check if lower bound for true label is greater than upper bound for other labels
    for i in range(10):
        if i != true_label:
            if lower_bound[0][true_label] <= upper_bound[0][i]:
                res = False
                break


    return res

def back_substitute(lower, upper):
    curr_lower = lower[-1] # shape: (L_layer_out_dim, L_layer_in_dim + 1)
    curr_upper = upper[-1]
    for i in range(len(lower)-2, -1, -1):
        
        prev_lower = lower[i] # shape: (L-1_layer_out_dim, L-1_layer_in_dim + 1)
        prev_upper = upper[i]

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

        # print(prev_lower.shape, lower_positive.shape)

        curr_lower = lower_positive @ prev_lower_append + lower_negative @ prev_upper_append
        curr_upper = upper_positive @ prev_upper_append + upper_negative @ prev_lower_append

    return curr_lower, curr_upper
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
    parser.add_argument("--test", type=int, required=True, help="simple: 0 or complex: 1")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    if dataset == "mnist":
        in_ch, in_dim, num_class = 1, 28, 10
    elif dataset == "cifar10":
        in_ch, in_dim, num_class = 3, 32, 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if args.test == 0:
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
