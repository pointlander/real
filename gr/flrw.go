// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import "math"

// ScaleFactor returns a(t), ȧ(t), and ä(t).
type ScaleFactor func(t float64) (a, adot, addot float64)

// FLRW is a Friedmann–Lemaître–Robertson–Walker geometry
//
//	ds² = −dt² + a(t)² [ dr²/(1 − k r²) + r² dθ² + r² sin²θ dφ² ]
//
// in coordinates (t, r, θ, φ). K is the curvature constant k.
type FLRW struct {
	K     float64
	Scale ScaleFactor
}

func (f FLRW) Name() string { return "FLRW" }

func (f FLRW) CoordNames() [4]string { return [4]string{"t", "r", "theta", "phi"} }

func (f FLRW) factors(x Vec) (a, adot, r, s, c, kr float64) {
	t, r := x[0], x[1]
	th := x[2]
	s = math.Sin(th)
	c = math.Cos(th)
	a, adot, _ = f.Scale(t)
	kr = 1 - f.K*r*r
	return a, adot, r, s, c, kr
}

func (f FLRW) Cov(x Vec) Mat {
	a, _, r, s, _, kr := f.factors(x)
	a2 := a * a
	var g Mat
	g[0][0] = -1
	g[1][1] = a2 / kr
	g[2][2] = a2 * r * r
	g[3][3] = a2 * r * r * s * s
	return g
}

func (f FLRW) DCov(x Vec) Tensor3 {
	a, adot, r, s, c, kr := f.factors(x)
	a2 := a * a
	var dg Tensor3
	// ∂_t
	dg[0][1][1] = 2 * a * adot / kr
	dg[0][2][2] = 2 * a * adot * r * r
	dg[0][3][3] = 2 * a * adot * r * r * s * s
	// ∂_r
	dg[1][1][1] = a2 * (2 * f.K * r) / (kr * kr)
	dg[1][2][2] = 2 * a2 * r
	dg[1][3][3] = 2 * a2 * r * s * s
	// ∂_θ
	dg[2][3][3] = a2 * r * r * 2 * s * c
	return dg
}

// EdSScale is the flat, matter-dominated scale factor normalized so that
// a(t0) = 1 and H(t0) = H0, with t0 = 2/(3 H0).
func EdSScale(H0 float64) ScaleFactor {
	t0 := 2 / (3 * H0)
	return func(t float64) (float64, float64, float64) {
		if t <= 0 {
			return 0, 0, 0
		}
		a := math.Pow(t/t0, 2.0/3.0)
		adot := (2.0 / 3.0) * a / t
		addot := adot * (-1.0 / 3.0) / t
		return a, adot, addot
	}
}

// Cosmology is a flat-or-curved Friedmann model with H0 = 1.
// The density parameters are Ωm, Ωr, ΩΛ, and Ωk, and should sum to 1
// so that H(a = 1) = 1. Ωk > 0 is an open universe.
type Cosmology struct {
	OmM, OmR, OmL, OmK float64
}

// H returns H(a)/H0.
func (c Cosmology) H(a float64) float64 {
	if a <= 0 {
		return math.Inf(1)
	}
	s := c.OmM/(a*a*a) + c.OmR/(a*a*a*a) + c.OmK/(a*a) + c.OmL
	if s < 0 {
		return math.NaN()
	}
	return math.Sqrt(s)
}

// Hz returns H(z)/H0.
func (c Cosmology) Hz(z float64) float64 {
	return c.H(1 / (1 + z))
}

// Age is the cosmic time from a = 0 to scale factor a, in units of 1/H0.
// A pure de Sitter universe has infinite age; Age then returns +Inf.
func (c Cosmology) Age(a float64) float64 {
	if c.OmM == 0 && c.OmR == 0 && c.OmK == 0 {
		return math.Inf(1)
	}
	if a <= 0 {
		return 0
	}
	// a = u² removes the matter-era singularity of dt = da / (a H).
	return Simpson(func(u float64) float64 {
		if u == 0 {
			return 0
		}
		return 2 / (u * c.H(u*u))
	}, 0, math.Sqrt(a), 4000)
}

// Lookback is the light-travel time from redshift z to today, in units of 1/H0.
func (c Cosmology) Lookback(z float64) float64 {
	if z <= 0 {
		return 0
	}
	return Simpson(func(zp float64) float64 {
		return 1 / ((1 + zp) * c.Hz(zp))
	}, 0, z, 2000)
}

// ComovingDistance is χ = ∫ dz / H(z), in units of c/H0.
func (c Cosmology) ComovingDistance(z float64) float64 {
	if z <= 0 {
		return 0
	}
	return Simpson(func(zp float64) float64 {
		return 1 / c.Hz(zp)
	}, 0, z, 2000)
}

// TransverseComoving is f_k(χ). Ωk = −k in units H0 = a0 = c = 1.
func (c Cosmology) TransverseComoving(z float64) float64 {
	chi := c.ComovingDistance(z)
	k := -c.OmK
	if math.Abs(k) < 1e-14 {
		return chi
	}
	s := math.Sqrt(math.Abs(k))
	if k > 0 {
		return math.Sin(s*chi) / s
	}
	return math.Sinh(s*chi) / s
}

// LuminosityDistance is d_L = (1+z) f_k(χ).
func (c Cosmology) LuminosityDistance(z float64) float64 {
	return (1 + z) * c.TransverseComoving(z)
}

// AngularDiameterDistance is d_A = f_k(χ) / (1+z).
func (c Cosmology) AngularDiameterDistance(z float64) float64 {
	return c.TransverseComoving(z) / (1 + z)
}

// Distances collects the usual background observables at redshift z.
type Distances struct {
	Z        float64
	H        float64
	Age      float64
	Lookback float64
	Comoving float64
	DM       float64
	DL       float64
	DA       float64
}

// Observe evaluates background distances at redshift z. Age is the age today.
func (c Cosmology) Observe(z float64) Distances {
	return Distances{
		Z:        z,
		H:        c.Hz(z),
		Age:      c.Age(1),
		Lookback: c.Lookback(z),
		Comoving: c.ComovingDistance(z),
		DM:       c.TransverseComoving(z),
		DL:       c.LuminosityDistance(z),
		DA:       c.AngularDiameterDistance(z),
	}
}

// Simpson integrates f on [a, b] with n panels. n is rounded up to an even number.
func Simpson(f func(float64) float64, a, b float64, n int) float64 {
	if n < 2 {
		n = 2
	}
	if n%2 == 1 {
		n++
	}
	h := (b - a) / float64(n)
	s := f(a) + f(b)
	for i := 1; i < n; i++ {
		x := a + float64(i)*h
		if i%2 == 0 {
			s += 2 * f(x)
		} else {
			s += 4 * f(x)
		}
	}
	return s * h / 3
}
