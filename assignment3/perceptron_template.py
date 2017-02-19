"""
Template to fit a perceptron to MNIST data.
"""

from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
from sys import argv
from matplotlib import pyplot as plt
plt.style.use("ggplot")

IMG_SHAPE = [28, 28]                 # Shape of image.


def get_data():
    "Get data from file."
    val = pd.read_csv("validation.csv")
    test = pd.read_csv("test.csv")
    return dict(val=val, test=test)


def show_one(row):
    "Just for fun. If you want, you can use this to display a single image."
    label = row["label"]
    img = row.drop("label")
    pixels = np.reshape(img.values, IMG_SHAPE)
    fig, ax = plt.subplots()
    ax.imshow(pixels, cmap="bone_r")
    ax.set_title("Label: {0}".format(label))
    plt.show()


def linear_kernel(u, v):
    """
    Computes a the linear kernel K(u, v) = u dot v + 1
    """
    # TODO: Implement the linear kernel function.
    pass


def make_polynomial_kernel(d):
    """
    Returns a function which computes the degree-d polynomial kernel between a
    pair of vectors (represented as 1-d Numpy arrays) u and v.
    """
    def kernel(u, v):
        # TODO: Implement the polynomial kernel function.
        pass

    return kernel


def exponential_kernel(u, v):
    """
    Computes the exponential kernel between vectors u and v, with sigma = 10.
    """
    sigma = 10
    # TODO: Implement the exponential kernel function.
    pass



def compute_y_hat(x_t, y_mistake, X_mistake, kernel):
    """
    x_t is a vector representing the current training instance.
    y_mistake is a vector of the outputs for all points that the algorithm has
        gotten wrong so far.
    X_mistake is a matrix whose ith row is the ith input that the algorithm got
        wrong so far.
    kernel takes two vectors u, vand returns K(u, v).
    """
    def sign(x): return 1 if x >= 0 else -1
    n_mistake = len(y_mistake)
    if not n_mistake:
        return sign(0)
    else:
        # TODO: Compute y hat.
        pass


def compute_loss(m):
    """
    Given a boolean mistake vector, compute the losses for T = 100, 200, ...,
    1000.
    """
    loss = np.cumsum(m) / np.arange(1, len(m) + 1)
    return pd.Series(loss[99::100], np.arange(100, 1100, 100))


def fit_perceptron(df, kernel):
    """
    Given dataset df and kernel function kernel, run the perceptron algorithm
    for a single pass over the data. Return the average loss every 100 steps.
    """
    y = df["label"].values
    X = df.drop("label", axis=1).values.astype(np.float)
    N, D = X.shape
    m = np.repeat(False, N)     # The mistake vector. Initially, no mistakes.
    for t in range(N):
        # TODO: Define the variables x_t, y_mistake, X_mistake that are needed
        # to compute y_hat.
        # Hint: m is a boolean vector. Its ith element is True if the model
        # made a mistake on x_i. If b is length-n boolean vector and x is a
        # length-N vector of any type, you can grab the elements of x, where b
        # is True with x[b]. Try it. Similarly, if X is an N-by-D matrix, you
        # can grab the rows of X where b is True with X[b].
        x_t = None              # Put in your values for these variables.
        y_t = None
        y_mistake = None
        X_mistake = None
        y_hat = compute_y_hat(x_t, y_mistake, X_mistake, kernel)
        if y_hat != y_t:
            m[t] = True         # Store the mistake if the model guessed wrong.
    loss = compute_loss(m)
    return loss


def run_linear_kernel():
    "Analysis for part 1 of question 4.4."
    data = get_data()
    val = data["val"]
    loss = fit_perceptron(val, linear_kernel)
    fig, ax = plt.subplots()
    loss.plot(ax=ax, marker=".")
    ax.set_title("Perceptron with linear kernel\nEvaluated on validation set")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction mistakes")
    fig.savefig("linear-kernel")


def run_polynomial_kernel():
    "Analysis for part 2 of question 4.4."
    data = get_data()
    val = data["val"]
    ds = [1, 3, 5, 7, 10, 15, 20]
    losses = [fit_perceptron(val, make_polynomial_kernel(d)) for d in ds]
    losses = pd.concat(losses, axis=1)
    losses.columns = ds
    final_losses = losses.iloc[-1]
    fig, ax = plt.subplots()
    final_losses.plot(ax=ax, marker=".", legend=False)
    ax.set_xlim(0, 21)
    ax.set_title("Perceptron with polynomial kernel\n" +
                 "Evaluated on validation set")
    ax.set_xticks(ds)
    ax.set_xlabel("Degree of kernel")
    ax.set_ylabel("Final fraction mistakes")
    fig.savefig("polynomial-kernel.png")


def run_poly_expon():
    "Analysis for part 3 of question 4.4."
    # TODO: Supply the best-performing degree for the polynomial kernel
    best_degree = None          # Put in your value.
    data = get_data()
    test = data["test"]
    kernel_poly = make_polynomial_kernel(best_degree)
    loss_poly = fit_perceptron(test, kernel_poly)
    loss_expon = fit_perceptron(test, exponential_kernel)
    losses = pd.DataFrame(dict(polynomial=loss_poly,
                               expon=loss_expon))
    losses = losses[["polynomial", "expon"]]
    fig, ax = plt.subplots()
    losses.plot(ax=ax, marker=".")
    ax.set_title("Perceptron with polynomial and exponential kernels\n" +
                 "Evaluated on test set")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction mistakes")
    fig.savefig("poly-expon-kernel.png")


def main():
    "Based on command line flag, run one of 3 question subparts"
    flag = argv[1]
    dispatch = dict(linear=run_linear_kernel,
                    poly=run_polynomial_kernel,
                    expon=run_poly_expon)
    if flag not in dispatch:
        print("Not a valid argument to the script.")
    dispatch[flag]()


if __name__ == "__main__":
    main()
