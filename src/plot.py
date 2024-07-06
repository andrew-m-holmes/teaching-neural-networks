def plotcontour():
    pass


def plotsurface3D(
    resolution: int = 10, endpoints: Tuple[float, float] = (-1, 1)
) -> None:

    aspace = torch.linspace(*endpoints, resolution)
    bspace = torch.linspace(*endpoints, resolution)
    A, B = torch.meshgrid(aspace, bspace, indexing="ij")
    Z = torch.ones((resolution, resolution))

    print("Landscape generating...")
    start = time.perf_counter()

    for i in range(resolution):
        for j in range(resolution):
            thetahats = genminimizer(thetas, deltas, etas, A[i][j], B[i][j])
            loss = evaluate(model, thetas, thetahats, lossfn, testloader, device=device)
            Z[i, j] = loss
            print(f"Iteration: {i * 10 + j}, loss: {loss:.4f}")

    end = time.perf_counter()
    print(f"Elasped time: {(end - start):.2f} seconds")

    a = A.numpy()
    b = B.numpy()
    z = Z.numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlabel("alpha")
    plt.ylabel("beta")
    ax.set_zlabel("loss")
    ax.plot_surface(a, b, z, cmap="viridis", alpha=0.5)
    plt.show()
