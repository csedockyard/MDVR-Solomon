import matplotlib.pyplot as plt

def plot_multi_depot_routes(results):

    plt.close('all')  # clean state
    plt.figure(figsize=(10, 7))

    colors = ['red', 'green', 'blue', 'purple', 'orange']

    for i, result in enumerate(results):

        depot = result["depot"]
        customers = result["customers"]
        vehicles = result["vehicles"]

        color = colors[i % len(colors)]

        # Plot customers
        plt.scatter(customers["XCOORD"], customers["YCOORD"],
                    color=color, label=f"Customers (Depot {i})")

        # Plot depot
        plt.scatter(depot["XCOORD"], depot["YCOORD"],
                    color='black', marker='s', s=100)

        # Plot EACH vehicle
        for v_idx, vehicle in enumerate(vehicles):

            x = []
            y = []

            for node in vehicle:
                if node == 0:
                    x.append(depot["XCOORD"])
                    y.append(depot["YCOORD"])
                else:
                    cust = customers.iloc[node - 1]
                    x.append(cust["XCOORD"])
                    y.append(cust["YCOORD"])

            plt.plot(x, y, linewidth=1)

    plt.title("Multi-Depot Vehicle Routing (ACO)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    plt.show()