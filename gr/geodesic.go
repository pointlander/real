// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import "math"

// Causal character of a curve, equal to g_μν u^μ u^ν.
const (
	Timelike  = -1.0
	Null      = 0.0
	Spacelike = 1.0
)

// State is a point and tangent on a geodesic. Lambda is the affine parameter
// (proper time for a timelike curve normalized to −1).
type State struct {
	Lambda float64
	X, V   Vec
}

// Constraint returns g_μν v^μ v^ν.
func Constraint(m Metric, x, v Vec) float64 {
	return Contract(m.Cov(x), v, v)
}

// IntegrateConfig controls geodesic integration.
type IntegrateConfig struct {
	T1       float64
	Atol     float64
	Rtol     float64
	H0       float64
	Hmin     float64
	Hmax     float64
	MaxSteps int
	// Project restores g(v, v) = Kind after each accepted step by
	// adjusting the t-component, for charts where that quadratic is non-degenerate.
	Project bool
	Kind    float64
	Stop    func(State) bool
	Observe func(State)
}

// DefaultIntegrate is accurate enough for orbits of a few dozen M.
func DefaultIntegrate() IntegrateConfig {
	return IntegrateConfig{
		T1:       1,
		Atol:     1e-12,
		Rtol:     1e-9,
		H0:       1e-2,
		Hmin:     1e-14,
		Hmax:     2,
		MaxSteps: 2_000_000,
		Kind:     Timelike,
	}
}

// Result is the end state of an integration.
type Result struct {
	State    State
	Steps    int
	Rejects  int
	Stopped  bool
	Finished bool
}

// IntegrateGeodesic integrates the geodesic equation
//
//	d²x^ρ/dλ² + Γ^ρ_μν (dx^μ/dλ)(dx^ν/dλ) = 0
//
// from λ = 0 to cfg.T1. Finished is false when the stepper fails
// (singular chart, step-size collapse, or step limit).
func IntegrateGeodesic(m Metric, x, v Vec, cfg IntegrateConfig) Result {
	if cfg.Atol == 0 {
		cfg.Atol = 1e-12
	}
	if cfg.Rtol == 0 {
		cfg.Rtol = 1e-9
	}
	if cfg.Hmin == 0 {
		cfg.Hmin = 1e-14
	}
	if cfg.MaxSteps == 0 {
		cfg.MaxSteps = 2_000_000
	}
	y0 := make([]float64, 8)
	for i := 0; i < 4; i++ {
		y0[i] = x[i]
		y0[i+4] = v[i]
	}
	rhs := func(t float64, y, dst []float64) bool {
		var xx, vv Vec
		for i := 0; i < 4; i++ {
			xx[i] = y[i]
			vv[i] = y[i+4]
		}
		gamma, ok := Christoffel(m, xx)
		if !ok {
			return false
		}
		for i := 0; i < 4; i++ {
			dst[i] = vv[i]
		}
		for rho := 0; rho < 4; rho++ {
			var s float64
			for mu := 0; mu < 4; mu++ {
				for nu := 0; nu < 4; nu++ {
					s += gamma[rho][mu][nu] * vv[mu] * vv[nu]
				}
			}
			dst[4+rho] = -s
			if math.IsNaN(dst[4+rho]) || math.IsInf(dst[4+rho], 0) {
				return false
			}
		}
		return true
	}
	var obs func(float64, []float64)
	if cfg.Observe != nil || cfg.Project {
		obs = func(t float64, y []float64) {
			if cfg.Project {
				var xx, vv Vec
				for i := 0; i < 4; i++ {
					xx[i] = y[i]
					vv[i] = y[i+4]
				}
				if pv, ok := projectTime(m, xx, vv, cfg.Kind); ok {
					for i := 0; i < 4; i++ {
						y[i+4] = pv[i]
					}
				}
			}
			if cfg.Observe != nil {
				var st State
				st.Lambda = t
				for i := 0; i < 4; i++ {
					st.X[i] = y[i]
					st.V[i] = y[i+4]
				}
				cfg.Observe(st)
			}
		}
	}
	var stop func(float64, []float64) bool
	if cfg.Stop != nil {
		stop = func(t float64, y []float64) bool {
			var st State
			st.Lambda = t
			for i := 0; i < 4; i++ {
				st.X[i] = y[i]
				st.V[i] = y[i+4]
			}
			return cfg.Stop(st)
		}
	}
	t, y, steps, rejects, stopped, ok := dopri5(rhs, 0, y0, cfg.T1, cfg.Atol, cfg.Rtol, cfg.H0, cfg.Hmin, cfg.Hmax, cfg.MaxSteps, stop, obs)
	var st State
	st.Lambda = t
	for i := 0; i < 4; i++ {
		st.X[i] = y[i]
		st.V[i] = y[i+4]
	}
	return Result{State: st, Steps: steps, Rejects: rejects, Stopped: stopped, Finished: ok}
}

// projectTime adjusts v^t so that g(v, v) = kind, keeping the other components.
func projectTime(m Metric, x, v Vec, kind float64) (Vec, bool) {
	g := m.Cov(x)
	A := g[0][0]
	B := 0.0
	for i := 1; i < 4; i++ {
		B += 2 * g[0][i] * v[i]
	}
	C := -kind
	for i := 1; i < 4; i++ {
		for j := 1; j < 4; j++ {
			C += g[i][j] * v[i] * v[j]
		}
	}
	vt, ok := solveQuadratic(A, B, C, v[0])
	if !ok {
		return v, false
	}
	v[0] = vt
	return v, true
}

func solveQuadratic(a, b, c, prefer float64) (float64, bool) {
	scale := math.Max(1, math.Abs(b)+math.Abs(c))
	if math.Abs(a) < 1e-14*scale {
		if math.Abs(b) < 1e-18*scale {
			return 0, false
		}
		return -c / b, true
	}
	disc := b*b - 4*a*c
	if disc < 0 {
		return 0, false
	}
	s := math.Sqrt(disc)
	q := -0.5 * (b + math.Copysign(s, b))
	var r1, r2 float64
	switch {
	case q == 0 && b == 0:
		r1, r2 = 0, 0
	case q == 0:
		r1 = -b / a
		r2 = 0
	default:
		r1 = q / a
		r2 = c / q
	}
	if math.IsNaN(r1) || math.IsNaN(r2) {
		return 0, false
	}
	if math.Abs(r1-prefer) <= math.Abs(r2-prefer) {
		return r1, true
	}
	return r2, true
}

// ConstantsOfMotion returns E = −u_t and L = u_φ for a stationary axisymmetric chart
// whose coordinates are (t, r, θ, φ).
func ConstantsOfMotion(m Metric, x, v Vec) (E, L float64) {
	g := m.Cov(x)
	var ut, up float64
	for nu := 0; nu < 4; nu++ {
		ut += g[0][nu] * v[nu]
		up += g[3][nu] * v[nu]
	}
	return -ut, up
}

// VelocityFromConserved builds an equatorial tangent from Killing constants
// E = −u_t and L = u_φ. Coordinates must be (t, r, θ, φ).
// inbound selects the sign of u^r. kind is Timelike, Null, or Spacelike.
func VelocityFromConserved(m Metric, x Vec, E, L, kind float64, inbound bool) (Vec, bool) {
	g := m.Cov(x)
	a := g[0][0]
	b := g[0][3]
	c := g[3][3]
	det := a*c - b*b
	if math.Abs(det) < 1e-18 {
		return Vec{}, false
	}
	ut := ((-E)*c - b*L) / det
	up := (a*L - b*(-E)) / det
	quad := -E*ut + L*up
	if math.Abs(g[1][1]) < 1e-18 {
		return Vec{}, false
	}
	ur2 := (kind - quad) / g[1][1]
	if ur2 < -1e-10 {
		return Vec{}, false
	}
	if ur2 < 0 {
		ur2 = 0
	}
	ur := math.Sqrt(ur2)
	if inbound {
		ur = -ur
	}
	return Vec{ut, ur, 0, up}, true
}

// NormalizeTime rescales a mostly-timelike tangent to g(v, v) = kind
// by changing only v^t.
func NormalizeTime(m Metric, x, v Vec, kind float64) (Vec, bool) {
	return projectTime(m, x, v, kind)
}
