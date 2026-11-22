# real

A general-relativity simulator in Go. Units are geometric, G = c = 1, and the metric signature is (−, +, +, +). The module is `github.com/pointlander/real`. It has no third-party dependencies and needs Go 1.27.

Black-hole coordinates are (t, r, θ, φ). The ingoing Eddington–Finkelstein chart is (v, r, θ, φ). Minkowski coordinates are Cartesian (t, x, y, z). Angles are radians. With `--mass 1`, a length of 1 is one solar mass: about 1.477 km, or 4.925 μs.

Christoffel symbols are exact when a metric supplies its first derivatives. The Riemann tensor differentiates those symbols numerically. Timelike and null geodesics use an adaptive Dormand–Prince 5(4) integrator.

## Run

```sh
go test ./...
go run . demo
go run . help
```

`go build` writes a `real` binary in this directory. That binary is gitignored.

## Commands

```text
real demo
real curvature   --metric schwarzschild --mass 1 --r 10
real christoffel --metric kerr --mass 1 --spin 0.6 --r 8
real orbit       --mass 1 --peri 20 --apo 36
real deflect     --mass 1 --impact 15 --r 200
real infall      --mass 1 --r 10
real drag        --mass 1 --spin 0.8 --r0 25 --r1 8
real isco        --metric kerr --mass 1 --spin 0.9
real cosmo       --om 0.3 --ol 0.7 --z 1
real geodesic    --metric schwarzschild --x 0,10,1.5708,0 --v 1.2,0,0,0.04 --lambda 100
real xor 0 1
real cnot 0.7071,0 0,0 0.7071,0 0,0
```

| Command | What it reports |
| --- | --- |
| `demo` | Curvature, orbits, light deflection, infall, frame dragging, and a flat ΛCDM example |
| `curvature` | Kretschmann scalar, Ricci scalar, and the size of the Ricci and Einstein tensors |
| `christoffel` | Nonzero connection components |
| `orbit` | Periapsis precession of a Schwarzschild bound orbit, plus an equatorial plot |
| `deflect` | Deflection of a null ray, compared with a quadrature and with 4M/b |
| `infall` | Proper time for radial rest-fall in ingoing Eddington–Finkelstein coordinates |
| `drag` | Azimuth change between two radii, including frame dragging when the spin is nonzero |
| `isco` | Innermost stable circular orbits and the prograde photon orbit |
| `cosmo` | Age, lookback, and distances for an FLRW cosmology with H0 = 1 |
| `geodesic` | One geodesic. `--v` is required |
| `xor` | Classical XOR from half-turns of a circular geodesic |
| `cnot` | Quantum CNOT from the same half-turns, acting on amplitudes |

`orbit`, `deflect`, `infall`, and `geodesic` take `--out path.csv`. Shared metric flags are `--metric`, `--mass`, `--spin` (a = J/M), and `--charge`.

Metrics: `minkowski`, `schwarzschild`, `kerr`, `rn`, `kn`, `ef`.

## Library

Package `gr` is the simulator. The command above is a client of it.

```go
m := gr.Schwarzschild(1)
cur, ok := gr.CurvatureAt(m, gr.Vec{0, 10, math.Pi / 2, 0}, 1e-4)
if !ok {
    // singular chart, or the metric inverse failed
}
fmt.Println(cur.Kretschmann)
```

Metrics implement `Cov` and, when the derivatives are analytic, `DCov`. `FuncMetric` accepts a user-supplied component function and differentiates it numerically. Curvature helpers include `Christoffel`, `Riemann`, `Ricci`, `Einstein`, and `KretschmannScalar`. Geodesics go through `IntegrateGeodesic`. Orbit helpers include `CircularOrbits`, `ISCO`, `BoundOrbit`, `LightDeflection`, `AzimuthChange`, `RadialInfall`, and `StaticRedshift`. Cosmology is `Cosmology` with `H0 = 1`.

## XOR and CNOT

Both gates fly the prograde timelike circle at r = 10, M = 1. A false bit is a geodesic of zero length. A true bit advances φ by half a turn. The bit is 1 when the final azimuth is nearer π than a multiple of 2π.

`gr.XOR(a, b)` chains two of those steps. Zero true bits and two true bits both end near a multiple of 2π, so the readout is the exclusive or.

`gr.CNOT` does the same for each computational-basis amplitude of a two-qubit state, ordered `|00⟩`, `|01⟩`, `|10⟩`, `|11⟩`. The target bit chooses the starting azimuth, 0 or π. The control bit chooses whether that ship advances by half a turn. The amplitude is carried onto the azimuth it lands on, so a superposition stays coherent. The CLI example is `|+⟩|0⟩`, which lands on `(|00⟩ + |11⟩) / √2`.

```sh
go run . xor 1 0
go run . cnot 0.7071,0 0,0 0.7071,0 0,0
```

Each CNOT amplitude is `re` or `re,im`.

## License

BSD 3-clause. See [LICENSE](LICENSE).
