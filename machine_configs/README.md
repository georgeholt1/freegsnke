# Machine configurations

Here we store a number of **machine description** files in JSON format (`.json`) that describe a particular **tokamak geometry** in FreeGSNKE. Each machine configuration includes individual component JSON files (active coils, passive coils, limiter, wall, and magnetic probes) as well as a unified `machine.json` bundle that combines all components into a single file. A use-case for each of the machine description files (except "test") can be found in the `examples` directory.  

| Directory | What machine is this? | Source
| ------ | ------ | ------ |
| test | A test tokamak used in the FreeGSNKE unit tests. | N\A
| example | A simple toy tokamak. | Example0 notebook
| MAST-U | A **MAST-U-like** tokamak. | UKAEA
| SPARC | A **SPARC-U-like** tokamak. | [SPARCPublic](https://github.com/cfs-energy/SPARCPublic)
| ITER | An **ITER-like** tokamak. | [FUSE.jl](https://github.com/ProjectTorreyPines/FUSE.jl)

