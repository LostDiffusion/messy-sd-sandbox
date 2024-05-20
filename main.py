#!/bin/python
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    print("[+] Starting... \n")

    tensors = {
        "weight1": torch.zeros((1024)),
        "weight2": torch.zeros((1024))
    }

    # Save the file with the tensors we just created.
    save_file(tensors, "model.safetensors")

    # Clear the current tensor variable
    tensors = {}

    # Read in the tensor we saved
    with safe_open("model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
        print("[+] Tensor read.")



if __name__ == "__main__":
    main()
