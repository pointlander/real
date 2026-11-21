// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import (
	"math"
	"testing"
)

func close(t *testing.T, got, want, tol float64, label string) {
	t.Helper()
	if math.IsNaN(got) || math.IsNaN(want) || math.Abs(got-want) > tol {
		t.Fatalf("%s: got %.12g want %.12g (tol %g)", label, got, want, tol)
	}
}

func TestInverse(t *testing.T) {
	m := Mat{
		{2, 1, 0, 0},
		{1, 3, 1, 0},
		{0, 1, 4, 1},
		{0, 0, 1, 5},
	}
	inv, ok := m.Inv()
	if !ok {
		t.Fatal("invert")
	}
	p := MulMat(m, inv)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			want := 0.0
			if i == j {
				want = 1
			}
			close(t, p[i][j], want, 1e-12, "I")
		}
	}
}

func TestMinkowski(t *testing.T) {
	m := Minkowski{}
	x := Vec{0, 1, 2, 3}
	g, ok := Christoffel(m, x)
	if !ok {
		t.Fatal("christoffel")
	}
	if MaxAbs3(g) != 0 {
		t.Fatalf("flat connection %g", MaxAbs3(g))
	}
	cur, ok := CurvatureAt(m, x, 1e-4)
	if !ok {
		t.Fatal("curvature")
	}
	close(t, cur.Kretschmann, 0, 0, "K")
	close(t, cur.Scalar, 0, 0, "R")

	v := Vec{math.Sqrt(1 + 0.01 + 0.04 + 0.09), 0.1, -0.2, 0.3}
	close(t, Constraint(m, x, v), -1, 1e-12, "norm")
	cfg := DefaultIntegrate()
	cfg.T1 = 10
	cfg.Hmax = 1
	res := IntegrateGeodesic(m, x, v, cfg)
	if !res.Finished {
		t.Fatal("geodesic failed")
	}
	for i := 0; i < 4; i++ {
		close(t, res.State.X[i], x[i]+10*v[i], 1e-8, "x")
		close(t, res.State.V[i], v[i], 1e-8, "v")
	}
}

func TestDopriExponential(t *testing.T) {
	f := func(t float64, y, dst []float64) bool {
		dst[0] = y[0]
		return true
	}
	t1, y, _, _, _, ok := dopri5(f, 0, []float64{1}, 1, 1e-12, 1e-10, 0.1, 1e-14, 1, 10000, nil, nil)
	if !ok {
		t.Fatal("stepper failed")
	}
	close(t, t1, 1, 0, "t")
	close(t, y[0], math.E, 1e-10, "exp")
}

func TestSchwarzschildConnectionAndCurvature(t *testing.T) {
	m := Schwarzschild(1)
	x := Vec{0, 10, math.Pi / 2, 0}
	g, ok := Christoffel(m, x)
	if !ok {
		t.Fatal("christoffel")
	}
	close(t, g[0][0][1], 1.0/(10*8), 1e-12, "Γ^t_tr")
	close(t, g[1][0][0], 1.0*(10-2)/1000, 1e-12, "Γ^r_tt")
	close(t, g[1][1][1], -1.0/(10*8), 1e-12, "Γ^r_rr")
	close(t, g[1][2][2], -8, 1e-12, "Γ^r_θθ")
	close(t, g[1][3][3], -8, 1e-12, "Γ^r_φφ")
	close(t, g[2][1][2], 0.1, 1e-12, "Γ^θ_rθ")
	close(t, g[3][1][3], 0.1, 1e-12, "Γ^φ_rφ")

	cur, ok := CurvatureAt(m, x, 1e-4)
	if !ok {
		t.Fatal("curvature")
	}
	close(t, cur.Kretschmann, SchwarzschildKretschmann(1, 10), 1e-10, "K")
	close(t, cur.Scalar, 0, 1e-9, "R")
	if cur.Ricci.MaxAbs() > 1e-8 {
		t.Fatalf("Ricci %g", cur.Ricci.MaxAbs())
	}
	if cur.Einstein.MaxAbs() > 1e-8 {
		t.Fatalf("Einstein %g", cur.Einstein.MaxAbs())
	}
	// R^r_{t r t} = −2M(r−2M)/r⁴.
	close(t, cur.Riemann[1][0][1][0], -2.0*(10-2)/10000, 1e-8, "R^r_trt")
	down := LowerRiemann(m.Cov(x), cur.Riemann)
	close(t, down[1][0][1][0], -2.0/1000, 1e-8, "R_rtrt")
	if math.Abs(cur.Riemann[1][0][1][0]+cur.Riemann[1][0][0][1]) > 1e-8 {
		t.Fatalf("antisymmetry %g %g", cur.Riemann[1][0][1][0], cur.Riemann[1][0][0][1])
	}
}

func TestKerrNewmanExactCurvature(t *testing.T) {
	cases := []KerrNewman{
		{M: 1, A: 0.6, Q: 0},
		{M: 1, A: 0, Q: 0.5},
		{M: 1, A: 0.4, Q: 0.3},
		{M: 1, A: 0, Q: 0},
	}
	angles := []float64{math.Pi / 2, math.Pi / 3, 0.4}
	for _, m := range cases {
		for _, th := range angles {
			x := Vec{0, 8, th, 0.2}
			// Derivatives agree with a central difference.
			got := m.DCov(x)
			want := NumericalDCov(m, x, 1e-6)
			for a := 0; a < 4; a++ {
				for b := 0; b < 4; b++ {
					for c := 0; c < 4; c++ {
						scale := math.Max(1, math.Abs(want[a][b][c]))
						if math.Abs(got[a][b][c]-want[a][b][c]) > 1e-6*scale {
							t.Fatalf("%s ∂g[%d%d%d] got %g want %g", m.Name(), a, b, c, got[a][b][c], want[a][b][c])
						}
					}
				}
			}
			g := m.Cov(x)
			ginv, ok := g.Inv()
			if !ok {
				t.Fatal("inv")
			}
			id := MulMat(g, ginv)
			for i := 0; i < 4; i++ {
				for j := 0; j < 4; j++ {
					want := 0.0
					if i == j {
						want = 1
					}
					close(t, id[i][j], want, 1e-9, "g ginv")
				}
			}
			s := math.Sin(th)
			sig, delta, _, _, _ := m.sigmaDelta(x[1], th)
			_ = sig
			detTP := g[0][0]*g[3][3] - g[0][3]*g[0][3]
			close(t, detTP, -delta*s*s, 1e-9, "tφ det")

			cur, ok := CurvatureAt(m, x, 1e-4)
			if !ok {
				t.Fatal("curvature")
			}
			close(t, cur.Kretschmann, m.KretschmannExact(8, th), 5e-8*math.Max(1, math.Abs(m.KretschmannExact(8, th))), "K")
			close(t, cur.Scalar, 0, 1e-7, "scalar")
			if m.Q == 0 && cur.Ricci.MaxAbs() > 1e-7 {
				t.Fatalf("vacuum Ricci %g for %s", cur.Ricci.MaxAbs(), m.Name())
			}
			if m.Q != 0 && cur.Ricci.MaxAbs() < 1e-6 {
				t.Fatalf("charged Ricci vanished for %s", m.Name())
			}
		}
	}
}

func TestFlatSphericalLimit(t *testing.T) {
	m := Schwarzschild(0)
	cur, ok := CurvatureAt(m, Vec{0, 5, math.Pi / 2, 0}, 1e-4)
	if !ok {
		t.Fatal("curvature")
	}
	if math.Abs(cur.Kretschmann) > 1e-8 || math.Abs(cur.Scalar) > 1e-8 {
		t.Fatalf("M=0 curvature K=%g R=%g", cur.Kretschmann, cur.Scalar)
	}
}

func TestCircularOrbitAndISCO(t *testing.T) {
	m := Schwarzschild(1)
	pos, _, pok, _ := CircularOrbits(m, 10, math.Pi/2)
	if !pok {
		t.Fatal("circular")
	}
	omega := math.Sqrt(1 / 1000.0)
	close(t, pos.Omega, omega, 1e-12, "Ω")
	close(t, pos.V[0], 1/math.Sqrt(1-0.3), 1e-12, "ut")
	close(t, Constraint(m, pos.X, pos.V), -1, 1e-12, "norm")

	cfg := DefaultIntegrate()
	period := 2 * math.Pi / pos.V[3]
	cfg.T1 = period
	cfg.Hmax = 0.5
	cfg.Rtol = 1e-10
	res := IntegrateGeodesic(m, pos.X, pos.V, cfg)
	if !res.Finished {
		t.Fatal("orbit failed")
	}
	close(t, res.State.X[1], 10, 1e-6, "r")
	close(t, res.State.X[3], 2*math.Pi, 1e-6, "φ")
	E0, L0 := ConstantsOfMotion(m, pos.X, pos.V)
	E1, L1 := ConstantsOfMotion(m, res.State.X, res.State.V)
	close(t, E1, E0, 1e-8, "E")
	close(t, L1, L0, 1e-8, "L")
	close(t, Constraint(m, res.State.X, res.State.V), -1, 1e-8, "drift")

	r, ok := ISCO(m, 3.2, 40, true)
	if !ok {
		t.Fatal("isco")
	}
	close(t, r, 6, 1e-4, "ISCO")
	ph, ok := PhotonOrbit(m, 2.2, 8, true)
	if !ok {
		t.Fatal("photon")
	}
	close(t, ph, 3, 1e-6, "photon sphere")
	b, ok := CriticalImpact(m, ph, true)
	if !ok {
		t.Fatal("b")
	}
	close(t, b, 3*math.Sqrt(3), 1e-5, "shadow")

	k := Kerr(1, 0.9)
	want, ok := KerrISCORadius(1, 0.9, true)
	if !ok {
		t.Fatal("analytic isco")
	}
	got, ok := ISCO(k, 1.5, 20, true)
	if !ok {
		t.Fatal("kerr isco")
	}
	close(t, got, want, 2e-3, "kerr prograde ISCO")
	wantR, ok := KerrISCORadius(1, 0.9, false)
	if !ok {
		t.Fatal("analytic retro")
	}
	gotR, ok := ISCO(k, 2, 20, false)
	if !ok {
		t.Fatal("kerr retro isco")
	}
	close(t, gotR, wantR, 2e-3, "kerr retrograde ISCO")
	if !(got < 6 && gotR > 6) {
		t.Fatalf("spin splitting %g %g", got, gotR)
	}
}

func TestPrecession(t *testing.T) {
	m := Schwarzschild(1)
	rp, ra := 40.0, 70.0
	E, L, ok := TurningPointConstants(m, rp, ra, math.Pi/2)
	if !ok {
		t.Fatal("EL")
	}
	want := SchwarzschildPrecession(1, E, L, rp, ra)
	ell := 2 * rp * ra / (rp + ra)
	weak := 6 * math.Pi / ell
	// The O(M/ℓ) term is several percent at these radii; the geodesic is
	// compared with the orbit-equation quadrature below.
	if math.Abs(want-weak)/weak > 0.15 {
		t.Fatalf("quadrature %g far from weak-field %g", want, weak)
	}
	cfg := DefaultIntegrate()
	cfg.T1 = 2e4
	cfg.Hmax = 1
	cfg.Rtol = 1e-9
	cfg.Atol = 1e-11
	orb := BoundOrbit(m, rp, ra, 1, cfg)
	if !orb.Finished || len(orb.PeriPhi) < 1 {
		t.Fatalf("orbit finished=%v peri=%d stepsamples=%d", orb.Finished, len(orb.PeriPhi), len(orb.Samples))
	}
	close(t, orb.Precession, want, 2e-3, "precession")
	close(t, orb.PeriR[0], rp, 1e-3, "peri r")
}

func TestLightDeflection(t *testing.T) {
	m := Schwarzschild(1)
	b, R := 25.0, 400.0
	want, rmin, ok := SchwarzschildDeflectionAngle(1, b, R)
	if !ok {
		t.Fatal("quad")
	}
	cfg := DefaultIntegrate()
	cfg.Hmax = 2
	cfg.Rtol = 1e-9
	cfg.T1 = 20 * R
	d := LightDeflection(m, b, R, 2.2, cfg)
	if !d.Finished || d.Captured {
		t.Fatalf("ray finished=%v captured=%v rmin=%g", d.Finished, d.Captured, d.RMin)
	}
	close(t, d.Angle, want, 5e-4, "deflection")
	close(t, d.RMin, rmin, 1e-3, "rmin")
	if math.Abs(d.Constraint) > 1e-6 {
		t.Fatalf("null constraint %g", d.Constraint)
	}
	// Weak-field infinity-to-infinity limit, loose because R is finite.
	if math.Abs(d.Angle-4/b)/(4/b) > 0.15 {
		t.Fatalf("not near 4M/b: %g vs %g", d.Angle, 4/b)
	}
	cap := LightDeflection(m, 4, 80, 2.2, cfg)
	if !cap.Captured {
		t.Fatalf("expected capture, rmin=%g φ=%g", cap.RMin, cap.DeltaPhi)
	}
}

func TestFrameDragging(t *testing.T) {
	flat := Schwarzschild(1)
	kerr := Kerr(1, 0.85)
	cfg := DefaultIntegrate()
	cfg.Hmax = 0.5
	cfg.Rtol = 1e-9
	cfg.T1 = 200
	g0, q0 := AzimuthChange(flat, 1, 0, 25, 8, cfg)
	if !g0.Finished || !q0.Finished {
		t.Fatal("schwarzschild drag failed")
	}
	close(t, g0.DeltaPhi, 0, 1e-6, "no drag")
	close(t, g0.DeltaPhi, q0.DeltaPhi, 1e-5, "quad match schw")

	g1, q1 := AzimuthChange(kerr, 1, 0, 25, 8, cfg)
	if !g1.Finished || !q1.Finished {
		t.Fatal("kerr drag failed")
	}
	if g1.DeltaPhi <= 0.01 {
		t.Fatalf("expected prograde frame dragging, Δφ=%g", g1.DeltaPhi)
	}
	close(t, g1.DeltaPhi, q1.DeltaPhi, 5e-4, "quad match kerr")
}

func TestInfall(t *testing.T) {
	M, r0 := 1.0, 10.0
	cfg := DefaultIntegrate()
	cfg.Hmax = 0.25
	cfg.Rtol = 1e-10
	cfg.Atol = 1e-12
	marks := []float64{2 * M, 1}
	inf := RadialInfall(M, r0, 0.4, marks, cfg)
	if !inf.Finished {
		t.Fatal("infall failed")
	}
	if math.Abs(inf.Constraint+1) > 1e-6 {
		t.Fatalf("constraint %g", inf.Constraint)
	}
	for i, r := range marks {
		want := RadialRestTau(M, r0, r)
		close(t, inf.TauAt[i], want, 2e-4, "tau")
	}
	x, v, ok := EFRestState(M, r0)
	if !ok {
		t.Fatal("state")
	}
	close(t, Constraint(SchwarzschildEF{M: M}, x, v), -1, 1e-12, "rest norm")
}

func TestRedshift(t *testing.T) {
	m := Schwarzschild(1)
	z, ok := StaticRedshift(m, 4, 1e6, math.Pi/2)
	if !ok {
		t.Fatal("z")
	}
	wantZ := math.Sqrt((1 - 2/1e6) / (1 - 2.0/4))
	close(t, z, wantZ, 1e-12, "1+z")
}

func TestFLRW(t *testing.T) {
	f := FLRW{K: 0, Scale: EdSScale(1)}
	t0 := 2.0 / 3.0
	x := Vec{t0, 0.4, math.Pi / 2, 0}
	g, ok := Christoffel(f, x)
	if !ok {
		t.Fatal("gamma")
	}
	close(t, g[1][0][1], 1, 1e-9, "Γ^r_tr = H")
	cur, ok := CurvatureAt(f, x, 1e-4)
	if !ok {
		t.Fatal("curvature")
	}
	close(t, cur.Scalar, 3, 1e-5, "R")
	close(t, cur.Einstein[0][0], 3, 2e-4, "G_tt")

	eds := Cosmology{OmM: 1}
	close(t, eds.Age(1), 2.0/3.0, 1e-10, "age")
	z := 1.0
	wantChi := 2 * (1 - 1/math.Sqrt(1+z))
	close(t, eds.ComovingDistance(z), wantChi, 1e-8, "chi")
	close(t, eds.LuminosityDistance(z), (1+z)*wantChi, 1e-8, "dL")
	close(t, eds.AngularDiameterDistance(z), wantChi/(1+z), 1e-8, "dA")
	lcdm := Cosmology{OmM: 0.3, OmL: 0.7}
	close(t, lcdm.H(1), 1, 1e-12, "H0")
	d := lcdm.Observe(1)
	if d.DL <= d.Comoving || d.Age <= 2.0/3.0 || d.DA <= 0 {
		t.Fatalf("lcdm distances %+v", d)
	}
}

func TestCustomMetric(t *testing.T) {
	// Schwarzschild supplied only as g_μν. Numerical derivatives must recover K.
	base := Schwarzschild(1)
	custom := FuncMetric{
		Title: "numeric-schwarzschild",
		Names: base.CoordNames(),
		G:     base.Cov,
		Step:  1e-5,
	}
	cur, ok := CurvatureAt(custom, Vec{0, 12, math.Pi / 2, 1}, 2e-4)
	if !ok {
		t.Fatal("curvature")
	}
	want := SchwarzschildKretschmann(1, 12)
	if math.Abs(cur.Kretschmann-want)/want > 2e-3 {
		t.Fatalf("numeric K got %g want %g", cur.Kretschmann, want)
	}
}

func TestProjectTime(t *testing.T) {
	m := Schwarzschild(1)
	x := Vec{0, 10, math.Pi / 2, 0}
	v := Vec{1.5, 0.02, -0.01, 0.03}
	got, ok := projectTime(m, x, v, Timelike)
	if !ok {
		t.Fatal("project")
	}
	close(t, Constraint(m, x, got), -1, 1e-12, "norm")
	close(t, got[1], v[1], 0, "ur")
	close(t, got[2], v[2], 0, "utheta")
	close(t, got[3], v[3], 0, "uphi")
	if got[0] <= 0 {
		t.Fatalf("future direction lost: %g", got[0])
	}
}
