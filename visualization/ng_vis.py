from __future__ import print_function

import argparse
import numpy as np

import neuroglancer
import neuroglancer.cli
import webbrowser


def add_example_layers(state):
    a = np.load(
        "/home/jakobtroidl/Desktop/NVP/reconstructed-nvp-25-t.npy"
    )

    b = a < 0.08160067
    b = b.astype(np.uint8) * 255

    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[1, 1, 1]
    )

    state.dimensions = dimensions
    state.layers.append(
        name="a",
        layer=neuroglancer.LocalVolume(
            data=a,
            dimensions=dimensions,
            voxel_offset=(1, 1, 1),
        ),
        shader="""
void main() {
  emitRGB(vec3(toNormalized(getDataValue(0)),
               toNormalized(getDataValue(1)),
               toNormalized(getDataValue(2))));
}
""",
    )
    state.layers.append(
        name="b",
        layer=neuroglancer.LocalVolume(
            data=b,
            dimensions=dimensions,
            volume_type="image",
        ),
    )
    return a, b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        a, b = add_example_layers(s)

    webbrowser.open_new(viewer.get_viewer_url())
