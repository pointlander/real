// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import "math"

// Metric is a Lorentzian metric g_μν on a coordinate chart.
// Implementations must be safe for concurrent evaluation.
type Metric interface {
	Name() string
	CoordNames() [4]string
	// Cov returns g_μν(x).
	Cov(x Vec) Mat
}

// DerivMetric supplies analytic first derivatives ∂_λ g_μν.
// Christoffel symbols built from these have no finite-difference truncation.
type DerivMetric interface {
	Metric
	// DCov returns ∂_λ g_μν as [λ][μ][ν].
	DCov(x Vec) Tensor3
}

// ConMetric supplies an analytic inverse g^μν.
type ConMetric interface {
	Metric
	Con(x Vec) (Mat, bool)
}

// FuncMetric is a user-supplied metric. Set D if analytic derivatives are known;
// otherwise derivatives are estimated by central differences.
type FuncMetric struct {
	Title string
	Names [4]string
	G     func(x Vec) Mat
	D     func(x Vec) Tensor3
	Step  float64
}

func (f FuncMetric) Name() string {
	if f.Title != "" {
		return f.Title
	}
	return "custom"
}

func (f FuncMetric) CoordNames() [4]string {
	if f.Names != ([4]string{}) {
		return f.Names
	}
	return [4]string{"x0", "x1", "x2", "x3"}
}

func (f FuncMetric) Cov(x Vec) Mat { return f.G(x) }

func (f FuncMetric) DCov(x Vec) Tensor3 {
	if f.D != nil {
		return f.D(x)
	}
	h := f.Step
	if h == 0 {
		h = 1e-6
	}
	return NumericalDCov(f, x, h)
}

// Inverse returns g^μν, using an analytic inverse when the metric provides one.
func Inverse(m Metric, x Vec) (Mat, bool) {
	if c, ok := m.(ConMetric); ok {
		return c.Con(x)
	}
	return m.Cov(x).Inv()
}

// NumericalDCov estimates ∂_λ g_μν with a central difference.
// h is a relative step: the absolute step in coordinate λ is h*max(1, |x^λ|).
func NumericalDCov(m Metric, x Vec, h float64) Tensor3 {
	var dg Tensor3
	for lam := 0; lam < 4; lam++ {
		hh := h * math.Max(1, math.Abs(x[lam]))
		xp, xm := x, x
		xp[lam] += hh
		xm[lam] -= hh
		gp := m.Cov(xp)
		gm := m.Cov(xm)
		inv := 1 / (2 * hh)
		for mu := 0; mu < 4; mu++ {
			for nu := 0; nu < 4; nu++ {
				dg[lam][mu][nu] = (gp[mu][nu] - gm[mu][nu]) * inv
			}
		}
	}
	return dg
}

// DCovOf returns ∂_λ g_μν, analytic when available.
func DCovOf(m Metric, x Vec, h float64) Tensor3 {
	if d, ok := m.(DerivMetric); ok {
		return d.DCov(x)
	}
	if h == 0 {
		h = 1e-6
	}
	return NumericalDCov(m, x, h)
}
