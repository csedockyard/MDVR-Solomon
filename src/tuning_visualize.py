import matplotlib.pyplot as plt

def plot_convergence(history, title="ACO Convergence"):
    plt.figure(figsize=(8, 5))
    plt.plot(history, linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("Best Distance")
    plt.title(title)

    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()