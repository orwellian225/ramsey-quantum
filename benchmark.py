import matplotlib.pyplot as plt
import ramsey_util as ru
import seaborn as sns
import polars as pl
import time

def benchmark_hamiltonian_construction():
    schema = { "graph_order": pl.Int32, "clique_iset_orders": pl.String, "time_seconds": pl.Float32 }
    data = {
            "graph_order": [],
            "clique_iset_orders": [],
            "time_seconds": []
    }

    for trial in range(5):
        for kl in [(x, y) for x in range(2, 4) for y in range(2, 4)]:
            for n in range(1, 8):
                print(f"\rtrial {trial}: R{kl} = {n} ", end="")
                start = time.perf_counter()
                _ = ru.generate_ramsey_hamiltonian(n, kl[0], kl[1])
                end = time.perf_counter()
                duration = end - start

                data["graph_order"].append(n)
                data["clique_iset_orders"].append(str(kl))
                data["time_seconds"].append(duration)

    print("")

    dataframe = pl.from_dict(schema=schema, data=data)
    dataframe = dataframe.with_columns(
        pl.col("graph_order"),
        pl.col("clique_iset_orders"),
        pl.col("time_seconds")
    )
    dataframe.write_csv("./raw_data/hamiltonian_construction_times.csv")

    sns.set_style("whitegrid")
    ax = sns.lineplot(data=dataframe.select(
        pl.col("graph_order").alias("Graph Order"),
        pl.col("clique_iset_orders").alias("(k,l)"),
        pl.col("time_seconds").alias("Time (ms)") * 1000.,
    ), x="Graph Order", y="Time (ms)", hue="(k,l)", estimator="mean", palette="Set2")
    _ = ax.set_title("Hamiltonian Construction runtime")
    plt.savefig("./visualizations/hamiltonian_construction_runtime_lineplot.pdf")


def main():
    benchmark_hamiltonian_construction()

if __name__ == "__main__":
    main()
