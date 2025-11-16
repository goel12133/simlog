import random
import math
from simlog.tracker import track


def generate_data(n: int, noise_std: float = 1.0):
    """
    Generate synthetic data from y = 3x + 5 + noise.
    """
    xs = [i for i in range(n)]
    ys = [3 * x + 5 + random.gauss(0, noise_std) for x in xs]
    return xs, ys


def mse(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


@track(project="demo_linear_fit")
def run_experiment(n_points: int, noise_std: float, slope_guess: float, intercept_guess: float):
    """
    Very dumb "fit": we don't optimise anything,
    we just evaluate how good a fixed guess is.
    SimLog logs the MSE and our parameters.
    """
    # 1) generate synthetic data
    xs, ys = generate_data(n_points, noise_std=noise_std)

    # 2) compute predictions using our "guessed" line
    y_pred = [slope_guess * x + intercept_guess for x in xs]

    # 3) compute error
    error = mse(ys, y_pred)

    # 4) here you'd normally save plots, etc.
    # we'll just pretend we did:
    artifacts = []

    # 5) return metrics + artifacts in the format SimLog expects
    return {
        "metrics": {"mse": error},
        "artifacts": artifacts,
    }


if __name__ == "__main__":
    # Try a few different guesses
    print("Running experiment 1...")
    run_experiment(n_points=50, noise_std=2.0, slope_guess=2.5, intercept_guess=4.0)

    print("Running experiment 2...")
    run_experiment(n_points=50, noise_std=2.0, slope_guess=3.0, intercept_guess=5.0)

    print("Running experiment 3...")
    run_experiment(n_points=50, noise_std=2.0, slope_guess=3.5, intercept_guess=6.0)

    print("Done. Check logged runs with `simlog runs`.")
