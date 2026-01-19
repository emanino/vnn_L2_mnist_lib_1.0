# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:28:45 2026

@author: Edoardo
"""

import os
import sys
import csv
import onnxruntime
import torchvision
import numpy as np

# deletes an existing folder (if it exists)
# then creates a new one from scratch
# Arguments:
# - path: folder path
def remake_dir(path):
    if os.path.exists(path):
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
        os.rmdir(path)
    os.makedirs(path)

# creates a VNN-LIB 1.0 file (list of text lines) according to a fixed template:
# Arguments:
# - n_in: number of inputs to the classifier
# - n_out: number of output scores from the classifier
# - bounds_in: numpy array of size [n_in,2] with the lower,upper input bounds
# - class_out: predicted class we are checking the robustness of
def vnnlib_template(n_in, n_out, bounds_in, class_out):
    
    assert len(bounds_in) == n_in
    assert class_out >= 0 and class_out < n_out
    assert all(bounds_in[:,0] <= bounds_in[:,1])
    
    lines = []
    
    # intro comment
    lines.append("; MNIST robustness to approximated L2-ball perturbations:")
    lines.append("; a toy VNN-COMP benchmark for the AAAI'26 tutorial.")
    lines.append("; Author: Edoardo Manino")
    lines.append("")
    
    # input variable declarations
    # (declare-const X_i Real)
    lines.append("; Input Variables")
    for i in range(n_in):
        lines.append("(declare-const X_" + str(i) + " Real)")
    lines.append("")
    
    # output variable declarations
    # (declare-const Y_i Real)
    lines.append("; Output Variables")
    for i in range(n_out):
        lines.append("(declare-const Y_" + str(i) + " Real)")
    lines.append("")
    
    # input constraints, e.g. 0.0 <= X_54 <= 1.0
    # (assert (<= X_54 1.0))
    # (assert (>= X_54 0.0))
    lines.append("; Input Constraints")
    for i in range(n_in):
        lines.append("(assert (<= X_" + str(i) + " " + str(bounds_in[i,0]) + "))")
        lines.append("(assert (>= X_" + str(i) + " " + str(bounds_in[i,1]) + "))")
    lines.append("")
    
    # output constraints (negated!)
    #(assert (or
    #    (>= Y_0 Y_c)
    #    (>= Y_1 Y_c)
    #        ...
    #    (>= Y_N Y_c)
    #))
    lines.append("; Output Constraints")
    lines.append("(assert (or")
    for i in range(n_out):
        if i == class_out:
            continue
        lines.append("  (>= Y_" + str(i) + " Y_" + str(class_out) + ")")
    lines.append("))")
    lines.append("")
    
    return lines

# create VNN-LIB 1.0 files given the following:
N = 15                  # number of verification instance
EPS = 0.05              # size of the inputs L2-ball
VNN_COMP_TIMEOUT = 100  # per-instance verification timeout

# generate N benchmark instances in VNN-LIB 1.0 format
# accepts a random seed as a command line argument (required)
if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python generate_properties.py <RANDOM_SEED>")
        sys.exit(1)
    
    remake_dir("vnnlib")
    
    # set the seed for reproducibility reasons
    # numpy random functions are used to select MNIST inputs
    seed_val = int(sys.argv[1])
    np.random.seed(seed=seed_val)
    
    # load MNIST test set
    mnist = torchvision.datasets.MNIST("./", train = False, download=True)
    
    # extract N random MNIST test images
    input_ids = np.random.choice(mnist.data.size()[0], size=N, replace=False)
    
    # load pre-trained MNIST network
    onnx_filename = "./onnx/mnist_net_256x2_L2_1568x1.onnx"
    session = onnxruntime.InferenceSession(onnx_filename)
    input_name = session.get_inputs()[0].name
    
    # generate N benchmark instances
    instance_data = []
    for i, img_id in enumerate(input_ids):
        
        # normalise input in [0,1]^784 and flatten it
        img = mnist.data[img_id] / 255
        x_img = np.ndarray.flatten(img.cpu().detach().numpy())
        x_eps = np.zeros(784)
        x = np.concatenate([x_img, x_eps]).astype(np.float32)
        x = np.reshape(x, [1, 784*2])
        
        # compute c = argmax{y = f(x)}
        y = session.run(None, {input_name: x})
        c = int(np.argmax(y))
        
        # copy input image
        bounds_in = np.column_stack([x[0],x[0]])
        
        # overwrite the second half with perturbation inputs
        # the hyperbox is approximately transformed into L2 ball by the ML model
        bounds_in[784:,0] = -EPS
        bounds_in[784:,1] = +EPS
        
        # write VNN-LIB file
        lines = vnnlib_template(784 * 2, 10, bounds_in, c)
        vnnlib_filename = "./vnnlib/instance_" + str(i) + ".vnnlib"
        with open(vnnlib_filename, "w") as f:
            f.writelines(line + "\n" for line in lines)
        
        instance = [onnx_filename, vnnlib_filename, VNN_COMP_TIMEOUT]
        instance_data.append(instance)
    
    # save the ONNX/VNN-LIB instance pairs in the required CSV file
    with open('instances.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(instance_data)
