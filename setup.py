from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension(
            "freegsnke._freegsnke_rs",
            path="Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
        )
    ]
)
