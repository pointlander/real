// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command real simulates curvature and geodesics in general relativity.
//
// Units are geometric, G = c = 1, with metric signature (−, +, +, +).
// Black-hole coordinates are (t, r, θ, φ), except the Eddington–Finkelstein
// chart used for infall, which is (v, r, θ, φ). A solar mass is about
// 1.477 km, or 4.925 microseconds, of geometric length and time.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/pointlander/real/gr"
)

func main() {
	if len(os.Args) < 2 {
		usage(os.Stdout)
		return
	}
	var err error
	switch os.Args[1] {
	case "help", "-h", "--help":
		usage(os.Stdout)
	case "demo":
		err = demo(os.Stdout)
	case "curvature":
		err = cmdCurvature(os.Args[2:])
	case "christoffel":
		err = cmdChristoffel(os.Args[2:])
	case "orbit":
		err = cmdOrbit(os.Args[2:])
	case "deflect":
		err = cmdDeflect(os.Args[2:])
	case "infall":
		err = cmdInfall(os.Args[2:])
	case "drag":
		err = cmdDrag(os.Args[2:])
	case "isco":
		err = cmdISCO(os.Args[2:])
	case "cosmo":
		err = cmdCosmo(os.Args[2:])
	case "geodesic":
		err = cmdGeodesic(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", os.Args[1])
		usage(os.Stderr)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func usage(w io.Writer) {
	fmt.Fprintf(w, `real — general relativity simulator (G = c = 1, signature −+++)

Usage:
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

Metrics: minkowski, schwarzschild, kerr, rn, kn, ef
  --mass, --spin (a = J/M), --charge
Coordinates are (t, r, θ, φ) except ef, which is ingoing Eddington–Finkelstein (v, r, θ, φ).

A length of 1 with --mass 1 is one solar mass when scaled by 1.477 km
(time: 4.925 μs). Angles are radians.
`)
}

type holeFlags struct {
	metric string
	mass   float64
	spin   float64
	charge float64
}

func (h *holeFlags) add(fs *flag.FlagSet) {
	fs.StringVar(&h.metric, "metric", "schwarzschild", "minkowski, schwarzschild, kerr, rn, kn, or ef")
	fs.Float64Var(&h.mass, "mass", 1, "mass M")
	fs.Float64Var(&h.spin, "spin", 0, "angular momentum per unit mass a")
	fs.Float64Var(&h.charge, "charge", 0, "electric charge Q")
}

func (h holeFlags) build() (gr.Metric, error) {
	switch strings.ToLower(h.metric) {
	case "minkowski", "flat":
		return gr.Minkowski{}, nil
	case "schwarzschild", "schw":
		return gr.Schwarzschild(h.mass), nil
	case "kerr":
		return gr.Kerr(h.mass, h.spin), nil
	case "rn", "reissner", "reissner-nordstrom":
		return gr.ReissnerNordstrom(h.mass, h.charge), nil
	case "kn", "kerr-newman":
		return gr.KerrNewman{M: h.mass, A: h.spin, Q: h.charge}, nil
	case "ef", "eddington", "eddington-finkelstein":
		return gr.SchwarzschildEF{M: h.mass}, nil
	default:
		return nil, fmt.Errorf("unknown metric %q", h.metric)
	}
}

func cmdCurvature(args []string) error {
	fs := flag.NewFlagSet("curvature", flag.ContinueOnError)
	var h holeFlags
	h.add(fs)
	r := fs.Float64("r", 10, "radial coordinate")
	th := fs.Float64("theta", math.Pi/2, "polar angle")
	phi := fs.Float64("phi", 0, "azimuth")
	t0 := fs.Float64("t", 0, "time coordinate")
	step := fs.Float64("step", 1e-4, "Richardson step for ∂Γ")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(args); err != nil {
		return err
	}
	m, err := h.build()
	if err != nil {
		return err
	}
	x := gr.Vec{*t0, *r, *th, *phi}
	cur, ok := gr.CurvatureAt(m, x, *step)
	if !ok {
		return fmt.Errorf("curvature failed at %v (singular chart?)", x)
	}
	fmt.Printf("%s\n", m.Name())
	fmt.Printf("x = (%.6g, %.6g, %.6g, %.6g)\n", x[0], x[1], x[2], x[3])
	fmt.Printf("Kretschmann  %.10g\n", cur.Kretschmann)
	if kn, ok := m.(gr.KerrNewman); ok {
		fmt.Printf("exact        %.10g\n", kn.KretschmannExact(*r, *th))
	}
	fmt.Printf("Ricci scalar %.6g\n", cur.Scalar)
	fmt.Printf("max |Ricci|  %.6g\n", cur.Ricci.MaxAbs())
	fmt.Printf("max |G_μν|   %.6g\n", cur.Einstein.MaxAbs())
	return nil
}

func cmdChristoffel(args []string) error {
	fs := flag.NewFlagSet("christoffel", flag.ContinueOnError)
	var h holeFlags
	h.add(fs)
	r := fs.Float64("r", 10, "radial coordinate")
	th := fs.Float64("theta", math.Pi/2, "polar angle")
	tol := fs.Float64("tol", 1e-12, "hide components smaller than this")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(args); err != nil {
		return err
	}
	m, err := h.build()
	if err != nil {
		return err
	}
	comps, ok := gr.ChristoffelComponents(m, gr.Vec{0, *r, *th, 0}, *tol)
	if !ok {
		return fmt.Errorf("Christoffel failed")
	}
	fmt.Printf("%s at r=%.6g θ=%.6g\n", m.Name(), *r, *th)
	for _, c := range comps {
		fmt.Printf("  %-16s % .10g\n", c.Name, c.Value)
	}
	return nil
}

func cmdOrbit(args []string) error {
	fs := flag.NewFlagSet("orbit", flag.ContinueOnError)
	mass := fs.Float64("mass", 1, "mass")
	peri := fs.Float64("peri", 20, "periapsis radius")
	apo := fs.Float64("apo", 36, "apoapsis radius")
	periods := fs.Int("periods", 1, "radial periods to integrate")
	out := fs.String("out", "", "CSV path")
	plot := fs.Bool("plot", true, "draw an equatorial plot")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(args); err != nil {
		return err
	}
	m := gr.Schwarzschild(*mass)
	cfg := gr.DefaultIntegrate()
	cfg.T1 = 2e4 * float64(*periods) * (*apo / 30)
	cfg.Hmax = math.Max(0.5, *peri/40)
	cfg.Rtol = 1e-9
	orb := gr.BoundOrbit(m, *peri, *apo, *periods, cfg)
	if len(orb.PeriPhi) < 1 {
		return fmt.Errorf("orbit did not return to periapsis (finished=%v, samples=%d)", orb.Finished, len(orb.Samples))
	}
	E, L := orb.E, orb.L
	exact := gr.SchwarzschildPrecession(*mass, E, L, *peri, *apo)
	ell := 2 * *peri * *apo / (*peri + *apo)
	weak := 6 * math.Pi * *mass / ell
	fmt.Printf("Schwarzschild bound orbit  M=%.6g  rp=%.6g  ra=%.6g\n", *mass, *peri, *apo)
	fmt.Printf("E = %.8g    L = %.8g\n", E, L)
	fmt.Printf("precession   % .8g rad/orbit   (%.4g°)\n", orb.Precession, orb.Precession*180/math.Pi)
	fmt.Printf("quadrature   % .8g rad/orbit\n", exact)
	fmt.Printf("weak field   % .8g rad/orbit   (6πM/ℓ, ℓ=%.6g)\n", weak, ell)
	if *plot {
		fmt.Println(gr.PlotEquatorial(decimate(orb.Samples, 2500), 2**mass, 73, 31))
	}
	if *out != "" {
		return writeSamples(*out, m, orb.Samples)
	}
	return nil
}

func cmdDeflect(args []string) error {
	fs := flag.NewFlagSet("deflect", flag.ContinueOnError)
	mass := fs.Float64("mass", 1, "mass")
	b := fs.Float64("impact", 15, "impact parameter L/E")
	R := fs.Float64("r", 200, "launch radius")
	out := fs.String("out", "", "CSV path")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(args); err != nil {
		return err
	}
	m := gr.Schwarzschild(*mass)
	cfg := gr.DefaultIntegrate()
	cfg.Hmax = math.Max(1, *R/200)
	cfg.Rtol = 1e-9
	cfg.T1 = 30 * *R
	d := gr.LightDeflection(m, *b, *R, 2.2**mass, cfg)
	fmt.Printf("Schwarzschild light ray  M=%.6g  b=%.6g  R=%.6g\n", *mass, *b, *R)
	if d.Captured {
		fmt.Printf("captured (closest approach %.6g)\n", d.RMin)
	} else if !d.Finished {
		return fmt.Errorf("ray integration failed at r=%.6g", d.RMin)
	} else {
		exact, rmin, ok := gr.SchwarzschildDeflectionAngle(*mass, *b, *R)
		fmt.Printf("deflection   %.8g rad   (%.4g°)\n", d.Angle, d.Angle*180/math.Pi)
		if ok {
			fmt.Printf("quadrature   %.8g rad\n", exact)
			fmt.Printf("closest      %.8g    (quadrature %.8g)\n", d.RMin, rmin)
		}
		fmt.Printf("4M/b         %.8g rad   (infinity-to-infinity, weak field)\n", 4**mass / *b)
		fmt.Printf("Δφ           %.8g    flat Δφ %.8g\n", d.DeltaPhi, d.FlatPhi)
	}
	if *out != "" {
		return writeSamples(*out, m, d.Samples)
	}
	return nil
}

func cmdInfall(args []string) error {
	fs := flag.NewFlagSet("infall", flag.ContinueOnError)
	mass := fs.Float64("mass", 1, "mass")
	r0 := fs.Float64("r", 10, "release radius")
	rmin := fs.Float64("rmin", 0.5, "stop radius")
	out := fs.String("out", "", "CSV path")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg := gr.DefaultIntegrate()
	cfg.Hmax = 0.25
	cfg.Rtol = 1e-10
	marks := []float64{2 * *mass, 1 * *mass, *rmin}
	inf := gr.RadialInfall(*mass, *r0, *rmin, marks, cfg)
	if !inf.Finished {
		return fmt.Errorf("infall integration failed")
	}
	fmt.Printf("Radial fall from rest at r=%.6g, Schwarzschild-EF, M=%.6g\n", *r0, *mass)
	fmt.Printf("final constraint g(u,u)+1 = %.3g\n", inf.Constraint+1)
	for i, r := range marks {
		fmt.Printf("τ(r=%.6g) = %.8g    exact %.8g\n", r, inf.TauAt[i], gr.RadialRestTau(*mass, *r0, r))
	}
	if *out != "" {
		return writeSamples(*out, gr.SchwarzschildEF{M: *mass}, inf.Samples)
	}
	return nil
}

func cmdDrag(args []string) error {
	fs := flag.NewFlagSet("drag", flag.ContinueOnError)
	var h holeFlags
	h.add(fs)
	r0 := fs.Float64("r0", 25, "start radius")
	r1 := fs.Float64("r1", 8, "stop radius")
	E := fs.Float64("E", 1, "conserved energy −u_t")
	L := fs.Float64("L", 0, "conserved angular momentum u_φ")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(args); err != nil {
		return err
	}
	m, err := h.build()
	if err != nil {
		return err
	}
	cfg := gr.DefaultIntegrate()
	cfg.Hmax = 0.5
	cfg.Rtol = 1e-9
	cfg.T1 = 100 * math.Abs(*r0-*r1)
	geo, quad := gr.AzimuthChange(m, *E, *L, *r0, *r1, cfg)
	if !geo.Finished {
		return fmt.Errorf("integration failed")
	}
	fmt.Printf("%s  E=%.6g  L=%.6g  r: %.6g → %.6g\n", m.Name(), *E, *L, *r0, *r1)
	fmt.Printf("Δφ geodesic    %.8g rad\n", geo.DeltaPhi)
	fmt.Printf("Δφ quadrature  %.8g rad\n", quad.DeltaPhi)
	return nil
}

func cmdISCO(args []string) error {
	fs := flag.NewFlagSet("isco", flag.ContinueOnError)
	var h holeFlags
	h.add(fs)
	fs.SetOutput(io.Discard)
	if err := fs.Parse(args); err != nil {
		return err
	}
	m, err := h.build()
	if err != nil {
		return err
	}
	kn, isKN := m.(gr.KerrNewman)
	rLo, rHi := 2.2*h.mass, 40*h.mass
	if isKN {
		if outer, _, ok := kn.Horizons(); ok {
			rLo = outer * 1.05
			fmt.Printf("%s  M=%.6g  a=%.6g  Q=%.6g\n", m.Name(), kn.M, kn.A, kn.Q)
			fmt.Printf("outer horizon  %.8g\n", outer)
			if ergo, ok := kn.ErgosphereRadius(math.Pi / 2); ok {
				fmt.Printf("equatorial ergosphere  %.8g\n", ergo)
			}
		}
	} else {
		fmt.Printf("%s\n", m.Name())
	}
	for _, branch := range []struct {
		name string
		pos  bool
	}{{"prograde", true}, {"retrograde", false}} {
		r, ok := gr.ISCO(m, rLo, rHi, branch.pos)
		if !ok {
			fmt.Printf("ISCO %-10s  (none)\n", branch.name)
			continue
		}
		fmt.Printf("ISCO %-10s  %.8g\n", branch.name, r)
		if isKN && kn.Q == 0 && math.Abs(kn.A) <= kn.M {
			if exact, ok := gr.KerrISCORadius(kn.M, kn.A, branch.pos); ok {
				fmt.Printf("  Bardeen formula  %.8g\n", exact)
			}
		}
	}
	ph, ok := gr.PhotonOrbit(m, rLo, math.Min(rHi, 12*h.mass), true)
	if ok {
		b, bok := gr.CriticalImpact(m, ph, true)
		fmt.Printf("prograde photon orbit  %.8g", ph)
		if bok {
			fmt.Printf("    b = L/E = %.8g", b)
		}
		fmt.Println()
	}
	return nil
}

func cmdCosmo(args []string) error {
	fs := flag.NewFlagSet("cosmo", flag.ContinueOnError)
	om := fs.Float64("om", 0.3, "Ωm")
	orad := fs.Float64("or", 0, "Ωr")
	ol := fs.Float64("ol", 0.7, "ΩΛ")
	ok := fs.Float64("ok", math.NaN(), "Ωk (default: 1−Ωm−Ωr−ΩΛ)")
	z := fs.Float64("z", 1, "redshift")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(args); err != nil {
		return err
	}
	omk := *ok
	if math.IsNaN(omk) {
		omk = 1 - *om - *orad - *ol
	}
	c := gr.Cosmology{OmM: *om, OmR: *orad, OmL: *ol, OmK: omk}
	d := c.Observe(*z)
	fmt.Printf("FLRW  H0 = 1   Ωm=%.6g  Ωr=%.6g  ΩΛ=%.6g  Ωk=%.6g\n", *om, *orad, *ol, omk)
	fmt.Printf("age today          %.8g / H0\n", d.Age)
	fmt.Printf("H(z)/H0            %.8g\n", d.H)
	fmt.Printf("lookback           %.8g / H0\n", d.Lookback)
	fmt.Printf("comoving χ         %.8g c/H0\n", d.Comoving)
	fmt.Printf("transverse d_M     %.8g c/H0\n", d.DM)
	fmt.Printf("luminosity d_L     %.8g c/H0\n", d.DL)
	fmt.Printf("angular d_A        %.8g c/H0\n", d.DA)
	return nil
}

func cmdGeodesic(args []string) error {
	fs := flag.NewFlagSet("geodesic", flag.ContinueOnError)
	var h holeFlags
	h.add(fs)
	xs := fs.String("x", "0,10,1.5707963267948966,0", "comma-separated coordinates")
	vs := fs.String("v", "", "comma-separated tangent; required")
	kind := fs.String("kind", "timelike", "timelike, null, or spacelike")
	norm := fs.Bool("normalize", false, "adjust v^t so g(v,v) matches --kind")
	lam := fs.Float64("lambda", 50, "affine-parameter limit")
	out := fs.String("out", "", "CSV path")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *vs == "" {
		return fmt.Errorf("--v is required (four comma-separated components)")
	}
	m, err := h.build()
	if err != nil {
		return err
	}
	x, err := parseVec(*xs)
	if err != nil {
		return fmt.Errorf("--x: %w", err)
	}
	v, err := parseVec(*vs)
	if err != nil {
		return fmt.Errorf("--v: %w", err)
	}
	var k float64
	switch strings.ToLower(*kind) {
	case "timelike":
		k = gr.Timelike
	case "null", "light":
		k = gr.Null
	case "spacelike":
		k = gr.Spacelike
	default:
		return fmt.Errorf("unknown kind %q", *kind)
	}
	if *norm {
		nv, ok := gr.NormalizeTime(m, x, v, k)
		if !ok {
			return fmt.Errorf("could not normalize the tangent")
		}
		v = nv
	}
	cfg := gr.DefaultIntegrate()
	cfg.T1 = *lam
	cfg.Kind = k
	cfg.Hmax = math.Max(0.5, math.Abs(*lam)/200)
	var samples []gr.OrbitSample
	cfg.Observe = func(s gr.State) {
		samples = append(samples, gr.OrbitSample{Lambda: s.Lambda, X: s.X, V: s.V})
	}
	res := gr.IntegrateGeodesic(m, x, v, cfg)
	if !res.Finished && !res.Stopped {
		return fmt.Errorf("integration failed at λ=%.6g, x=%v", res.State.Lambda, res.State.X)
	}
	fmt.Printf("%s  steps=%d  rejects=%d  λ=%.8g\n", m.Name(), res.Steps, res.Rejects, res.State.Lambda)
	fmt.Printf("x = %.8g %.8g %.8g %.8g\n", res.State.X[0], res.State.X[1], res.State.X[2], res.State.X[3])
	fmt.Printf("v = %.8g %.8g %.8g %.8g\n", res.State.V[0], res.State.V[1], res.State.V[2], res.State.V[3])
	fmt.Printf("g(v,v) = %.8g\n", gr.Constraint(m, res.State.X, res.State.V))
	E, L := gr.ConstantsOfMotion(m, res.State.X, res.State.V)
	fmt.Printf("E = %.8g    L = %.8g\n", E, L)
	if *out != "" {
		return writeSamples(*out, m, samples)
	}
	return nil
}

func demo(w io.Writer) error {
	fmt.Fprintln(w, "General relativity simulator")
	fmt.Fprintln(w, "geometric units G = c = 1, signature (− + + +)")
	fmt.Fprintln(w, "1 solar mass ≈ 1.477 km ≈ 4.925 μs")
	fmt.Fprintln(w)

	m := gr.Schwarzschild(1.0)
	x := gr.Vec{0, 10, math.Pi / 2, 0}
	cur, ok := gr.CurvatureAt(m, x, 1e-4)
	if !ok {
		return fmt.Errorf("curvature failed")
	}
	fmt.Fprintln(w, "Schwarzschild, M = 1, equatorial point r = 10")
	fmt.Fprintf(w, "  Kretschmann     %.8e    exact %.8e\n", cur.Kretschmann, gr.SchwarzschildKretschmann(1, 10))
	fmt.Fprintf(w, "  Ricci scalar    %.3g\n", cur.Scalar)
	fmt.Fprintf(w, "  max |Ricci|     %.3g\n", cur.Ricci.MaxAbs())
	ph, _ := gr.PhotonOrbit(m, 2.2, 8, true)
	b, _ := gr.CriticalImpact(m, ph, true)
	isco, _ := gr.ISCO(m, 3.2, 40, true)
	fmt.Fprintf(w, "  photon sphere   %.8g\n", ph)
	fmt.Fprintf(w, "  shadow b_crit   %.8g     (3√3 M = %.8g)\n", b, 3*math.Sqrt(3))
	fmt.Fprintf(w, "  ISCO            %.8g\n", isco)
	z, _ := gr.StaticRedshift(m, 4, 1e8, math.Pi/2)
	fmt.Fprintf(w, "  redshift r=4M → ∞    1+z = %.8g\n", z)
	fmt.Fprintln(w)

	rp, ra := 20.0, 36.0
	cfg := gr.DefaultIntegrate()
	cfg.T1 = 8000
	cfg.Hmax = 0.8
	cfg.Rtol = 1e-9
	orb := gr.BoundOrbit(m, rp, ra, 1, cfg)
	exact := gr.SchwarzschildPrecession(1, orb.E, orb.L, rp, ra)
	fmt.Fprintf(w, "Bound orbit  periapsis %.0f  apoapsis %.0f\n", rp, ra)
	fmt.Fprintf(w, "  E = %.6g   L = %.6g\n", orb.E, orb.L)
	fmt.Fprintf(w, "  precession  %.6g rad (%.3g°)\n", orb.Precession, orb.Precession*180/math.Pi)
	fmt.Fprintf(w, "  quadrature  %.6g rad\n", exact)
	if len(orb.Samples) > 0 {
		fmt.Fprintln(w, "  one radial period in the equatorial plane; + is the hole")
		fmt.Fprintln(w, gr.PlotEquatorial(decimate(orb.Samples, 2500), 2, 73, 31))
	}
	fmt.Fprintln(w)

	fmt.Fprintln(w, "Light deflection from r = 250")
	fmt.Fprintf(w, "  %8s %12s %12s %s\n", "b", "deflection", "4M/b", "fate")
	rayCfg := gr.DefaultIntegrate()
	rayCfg.Hmax = 1.5
	rayCfg.Rtol = 1e-8
	rayCfg.T1 = 8000
	for _, impact := range []float64{4, 8, 15, 30} {
		d := gr.LightDeflection(m, impact, 250, 2.2, rayCfg)
		fate := "escaped"
		ang := fmt.Sprintf("%10.5f", d.Angle)
		if d.Captured {
			fate = "captured"
			ang = "         —"
		} else if !d.Finished {
			fate = "failed"
		}
		fmt.Fprintf(w, "  %8.2f %12s %12.5f %s\n", impact, ang, 4/impact, fate)
	}
	fmt.Fprintln(w)

	inf := gr.RadialInfall(1, 10, 0.5, []float64{2, 1}, gr.IntegrateConfig{
		T1: 80, Atol: 1e-12, Rtol: 1e-10, H0: 1e-2, Hmin: 1e-14, Hmax: 0.2, MaxSteps: 2_000_000,
	})
	fmt.Fprintln(w, "Radial infall from rest at r = 10 (ingoing Eddington–Finkelstein)")
	fmt.Fprintf(w, "  τ to horizon   %.6g    exact %.6g\n", inf.TauAt[0], gr.RadialRestTau(1, 10, 2))
	fmt.Fprintf(w, "  τ to r = 1     %.6g    exact %.6g\n", inf.TauAt[1], gr.RadialRestTau(1, 10, 1))
	fmt.Fprintln(w)

	k := gr.Kerr(1, 0.9)
	outer, _, _ := k.Horizons()
	pro, _ := gr.ISCO(k, outer*1.05, 20, true)
	ret, _ := gr.ISCO(k, outer*1.05, 20, false)
	proEx, _ := gr.KerrISCORadius(1, 0.9, true)
	retEx, _ := gr.KerrISCORadius(1, 0.9, false)
	fmt.Fprintf(w, "Kerr, M = 1, a = 0.9\n")
	fmt.Fprintf(w, "  outer horizon       %.6g\n", outer)
	fmt.Fprintf(w, "  ISCO prograde       %.6g    formula %.6g\n", pro, proEx)
	fmt.Fprintf(w, "  ISCO retrograde     %.6g    formula %.6g\n", ret, retEx)
	dragCfg := gr.DefaultIntegrate()
	dragCfg.Hmax = 0.5
	dragCfg.Rtol = 1e-9
	dragCfg.T1 = 400
	geo, quad := gr.AzimuthChange(k, 1, 0, 25, 8, dragCfg)
	fmt.Fprintf(w, "  frame drag L=0, r=25→8    Δφ = %.6g rad (quadrature %.6g)\n", geo.DeltaPhi, quad.DeltaPhi)
	fmt.Fprintln(w)

	cosmo := gr.Cosmology{OmM: 0.3, OmL: 0.7}
	d := cosmo.Observe(1)
	fmt.Fprintln(w, "Flat ΛCDM, Ωm = 0.3, ΩΛ = 0.7, H0 = 1")
	fmt.Fprintf(w, "  age              %.6g / H0\n", d.Age)
	fmt.Fprintf(w, "  z = 1 lookback   %.6g / H0\n", d.Lookback)
	fmt.Fprintf(w, "  d_L(z=1)         %.6g c/H0\n", d.DL)
	fmt.Fprintf(w, "  d_A(z=1)         %.6g c/H0\n", d.DA)
	return nil
}

func decimate(s []gr.OrbitSample, max int) []gr.OrbitSample {
	if max < 2 || len(s) <= max {
		return s
	}
	out := make([]gr.OrbitSample, 0, max)
	stride := float64(len(s)-1) / float64(max-1)
	for i := 0; i < max; i++ {
		out = append(out, s[int(math.Round(float64(i)*stride))])
	}
	return out
}

func parseVec(s string) (gr.Vec, error) {
	parts := strings.Split(s, ",")
	if len(parts) != 4 {
		return gr.Vec{}, fmt.Errorf("need 4 comma-separated numbers, got %d", len(parts))
	}
	var v gr.Vec
	for i, p := range parts {
		x, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return gr.Vec{}, err
		}
		v[i] = x
	}
	return v, nil
}

func writeSamples(path string, m gr.Metric, samples []gr.OrbitSample) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	names := m.CoordNames()
	header := []string{"lambda"}
	for _, n := range names {
		header = append(header, n)
	}
	for _, n := range names {
		header = append(header, "v_"+n)
	}
	header = append(header, "constraint", "E", "L")
	if err := w.Write(header); err != nil {
		return err
	}
	rec := make([]string, 0, 14)
	for _, s := range samples {
		rec = rec[:0]
		rec = append(rec, strconv.FormatFloat(s.Lambda, 'g', 16, 64))
		for i := 0; i < 4; i++ {
			rec = append(rec, strconv.FormatFloat(s.X[i], 'g', 16, 64))
		}
		for i := 0; i < 4; i++ {
			rec = append(rec, strconv.FormatFloat(s.V[i], 'g', 16, 64))
		}
		c := gr.Constraint(m, s.X, s.V)
		E, L := gr.ConstantsOfMotion(m, s.X, s.V)
		rec = append(rec,
			strconv.FormatFloat(c, 'g', 16, 64),
			strconv.FormatFloat(E, 'g', 16, 64),
			strconv.FormatFloat(L, 'g', 16, 64),
		)
		if err := w.Write(rec); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}
