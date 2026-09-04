"""Benchmark: plain dask vs windowed arrays vs cached chunk arrays.

Scales particle count from 10 to 1,000,000 on ds_2d_left_agrid.zarr.
"""

import time as time_mod

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import parcels
import parcels._sgrid as sgrid


def make_fieldset(ds: xr.Dataset) -> parcels.FieldSet:
    """Build a FieldSet from the 2D left A-grid zarr dataset."""
    ds = ds.copy()
    ds["lon"].attrs["units"] = "m"
    ds["lat"].attrs["units"] = "m"
    ds = ds.pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.LOW),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "ZG", sgrid.Padding.LOW),),
        ),
    )
    return parcels.FieldSet.from_sgrid_conventions(
        ds,
        vector_fields={"UV": ("U_A_grid", "V_A_grid")},
        skip_field_data_validation=True,
    )


def delete_on_boundary(particles, fieldset):
    """Delete particles that hit the boundary instead of erroring."""
    particles.state = np.where(
        particles.state == parcels.StatusCode.ErrorOutOfBounds,
        parcels.StatusCode.Delete,
        particles.state,
    )


def run_simulation(fieldset, ds, npart):
    """Run a simulation and return elapsed time in seconds."""
    np.random.seed(42)
    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=parcels.Particle,
        t=np.full(npart, ds.time.values[0]),
        z=np.full(npart, 1),
        y=np.random.uniform(1.0, 5.0, npart),
        x=np.random.uniform(1.0, 5.0, npart),
    )

    t0 = time_mod.perf_counter()
    pset.execute(
        [parcels.kernels.AdvectionRK2, delete_on_boundary],
        runtime=np.timedelta64(100, "ms"),
        dt=np.timedelta64(10, "ms"),
    )
    return time_mod.perf_counter() - t0


def main():
    zarr_path = "./datasets/ds_2d_left_agrid.zarr"
    particle_counts = [10, 100, 1_000, 10_000, 100_000, 1_000_000]

    print(f"Loading dataset from {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=False)
    print(f"  shape: {dict(ds.dims)}")
    print(f"  chunks: U_A_grid {ds['U_A_grid'].encoding.get('chunks', 'N/A')}")

    # methods = {
    #     "plain dask": lambda ds: make_fieldset(ds),
    # }
    methods = {
        "windowed": lambda ds: make_fieldset(ds).to_windowed_arrays(),
        "cached chunks": lambda ds: make_fieldset(ds).to_chunk_cached_arrays(),
    }

    results = {name: [] for name in methods}

    for npart in particle_counts:
        print(f"\n--- {npart:,} particles ---")
        for name, build_fn in methods.items():
            fieldset = build_fn(ds)
            elapsed = run_simulation(fieldset, ds, npart)
            results[name].append(elapsed)
            print(f"  {name}: {elapsed:.3f}s")

    # --- Print results table ---
    print("\n" + "=" * 60)
    print("Results summary")
    print("=" * 60)
    header = f"{'N particles':>12s}"
    for name in methods:
        header += f"  {name:>15s}"
    print(header)
    print("-" * len(header))
    for i, npart in enumerate(particle_counts):
        row = f"{npart:>12,d}"
        for name in methods:
            row += f"  {results[name][i]:>14.3f}s"
        print(row)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "^", "D"]
    for j, (name, times) in enumerate(results.items()):
        ax.loglog(
            particle_counts,
            times,
            marker=markers[j % len(markers)],
            linewidth=2,
            markersize=7,
            label=name,
        )

    ax.set_xlabel("Number of particles")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Parcels simulation scaling: windowed vs cached chunk arrays")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("benchmark_chunk_cache.png", dpi=150)
    print("\nPlot saved to benchmark_chunk_cache.png")
    plt.show()


if __name__ == "__main__":
    main()
