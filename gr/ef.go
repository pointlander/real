// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import "math"

// SchwarzschildEF is the Schwarzschild geometry in ingoing Eddington–Finkelstein
// coordinates (v, r, θ, φ). The chart is regular on the future horizon, so a
// radial geodesic can be integrated from outside the hole through r = 2M.
type SchwarzschildEF struct {
	M float64
}

func (s SchwarzschildEF) Name() string { return "Schwarzschild-EF" }

func (s SchwarzschildEF) CoordNames() [4]string {
	return [4]string{"v", "r", "theta", "phi"}
}

func (s SchwarzschildEF) Cov(x Vec) Mat {
	r := x[1]
	th := x[2]
	sn := math.Sin(th)
	f := 1 - 2*s.M/r
	var g Mat
	g[0][0] = -f
	g[0][1] = 1
	g[1][0] = 1
	g[2][2] = r * r
	g[3][3] = r * r * sn * sn
	return g
}

func (s SchwarzschildEF) DCov(x Vec) Tensor3 {
	r := x[1]
	th := x[2]
	sn := math.Sin(th)
	cs := math.Cos(th)
	var dg Tensor3
	dg[1][0][0] = -2 * s.M / (r * r)
	dg[1][2][2] = 2 * r
	dg[1][3][3] = 2 * r * sn * sn
	dg[2][3][3] = 2 * r * r * sn * cs
	return dg
}

func (s SchwarzschildEF) Con(x Vec) (Mat, bool) {
	r := x[1]
	if r == 0 {
		return Mat{}, false
	}
	th := x[2]
	sn := math.Sin(th)
	if math.Abs(sn) < 1e-14 {
		return Mat{}, false
	}
	f := 1 - 2*s.M/r
	var g Mat
	g[0][1] = 1
	g[1][0] = 1
	g[1][1] = f
	g[2][2] = 1 / (r * r)
	g[3][3] = 1 / (r * r * sn * sn)
	return g, true
}

// RadialRestTau is the proper time for a radial Schwarzschild fall released
// from rest at r0, evaluated when the radius is r (0 ≤ r ≤ r0).
func RadialRestTau(M, r0, r float64) float64 {
	if M <= 0 || r0 <= 0 || r < 0 || r > r0 {
		return math.NaN()
	}
	cosEta := 2*r/r0 - 1
	if cosEta > 1 {
		cosEta = 1
	} else if cosEta < -1 {
		cosEta = -1
	}
	eta := math.Acos(cosEta)
	return math.Sqrt(r0*r0*r0/(8*M)) * (eta + math.Sin(eta))
}

// EFRestState is the initial state of a particle released from rest at radius r
// in ingoing Eddington–Finkelstein coordinates.
func EFRestState(M, r float64) (x, v Vec, ok bool) {
	if M <= 0 || r <= 2*M {
		return Vec{}, Vec{}, false
	}
	f := 1 - 2*M/r
	E := math.Sqrt(f)
	x = Vec{0, r, math.Pi / 2, 0}
	v = Vec{E / f, 0, 0, 0}
	return x, v, true
}
