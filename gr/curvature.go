// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import (
	"fmt"
	"math"
)

// Christoffel returns Γ^ρ_μν at x.
// The result is stored as [ρ][μ][ν].
func Christoffel(m Metric, x Vec) (Tensor3, bool) {
	ginv, ok := Inverse(m, x)
	if !ok {
		return Tensor3{}, false
	}
	dg := DCovOf(m, x, 1e-6)
	return christoffel(ginv, dg), true
}

func christoffel(ginv Mat, dg Tensor3) Tensor3 {
	var gamma Tensor3
	for rho := 0; rho < 4; rho++ {
		for mu := 0; mu < 4; mu++ {
			for nu := mu; nu < 4; nu++ {
				var s float64
				for sigma := 0; sigma < 4; sigma++ {
					term := dg[mu][nu][sigma] + dg[nu][mu][sigma] - dg[sigma][mu][nu]
					s += ginv[rho][sigma] * term
				}
				s *= 0.5
				gamma[rho][mu][nu] = s
				gamma[rho][nu][mu] = s
			}
		}
	}
	return gamma
}

// Riemann returns R^ρ_σμν at x, stored as [ρ][σ][μ][ν].
//
// The sign convention is
//
//	R^ρ_σμν = ∂_μ Γ^ρ_σν − ∂_ν Γ^ρ_σμ + Γ^ρ_μλ Γ^λ_σν − Γ^ρ_νλ Γ^λ_σμ.
//
// Christoffel symbols are differentiated with a Richardson central difference.
// h is the base relative step; zero selects the default (1e-4).
func Riemann(m Metric, x Vec, h float64) (Tensor4, bool) {
	if h == 0 {
		h = 1e-4
	}
	dG, ok := dChristoffel(m, x, h)
	if !ok {
		return Tensor4{}, false
	}
	gamma, ok := Christoffel(m, x)
	if !ok {
		return Tensor4{}, false
	}
	var R Tensor4
	for rho := 0; rho < 4; rho++ {
		for sigma := 0; sigma < 4; sigma++ {
			for mu := 0; mu < 4; mu++ {
				for nu := 0; nu < 4; nu++ {
					s := dG[mu][rho][sigma][nu] - dG[nu][rho][sigma][mu]
					for lam := 0; lam < 4; lam++ {
						s += gamma[rho][mu][lam]*gamma[lam][sigma][nu] - gamma[rho][nu][lam]*gamma[lam][sigma][mu]
					}
					R[rho][sigma][mu][nu] = s
				}
			}
		}
	}
	return R, true
}

// dChristoffel returns ∂_λ Γ^ρ_μν as [λ][ρ][μ][ν].
func dChristoffel(m Metric, x Vec, h float64) (Tensor4, bool) {
	fine, ok := dChristoffelRaw(m, x, h/2)
	if !ok {
		return Tensor4{}, false
	}
	coarse, ok := dChristoffelRaw(m, x, h)
	if !ok {
		return Tensor4{}, false
	}
	var out Tensor4
	for a := 0; a < 4; a++ {
		for b := 0; b < 4; b++ {
			for c := 0; c < 4; c++ {
				for d := 0; d < 4; d++ {
					out[a][b][c][d] = (4*fine[a][b][c][d] - coarse[a][b][c][d]) / 3
				}
			}
		}
	}
	return out, true
}

func dChristoffelRaw(m Metric, x Vec, h float64) (Tensor4, bool) {
	var out Tensor4
	for lam := 0; lam < 4; lam++ {
		hh := h * math.Max(1, math.Abs(x[lam]))
		xp, xm := x, x
		xp[lam] += hh
		xm[lam] -= hh
		gp, ok := Christoffel(m, xp)
		if !ok {
			return Tensor4{}, false
		}
		gm, ok := Christoffel(m, xm)
		if !ok {
			return Tensor4{}, false
		}
		inv := 1 / (2 * hh)
		for rho := 0; rho < 4; rho++ {
			for mu := 0; mu < 4; mu++ {
				for nu := 0; nu < 4; nu++ {
					out[lam][rho][mu][nu] = (gp[rho][mu][nu] - gm[rho][mu][nu]) * inv
				}
			}
		}
	}
	return out, true
}

// LowerRiemann returns R_ρσμν = g_ρλ R^λ_σμν.
func LowerRiemann(g Mat, R Tensor4) Tensor4 {
	var out Tensor4
	for a := 0; a < 4; a++ {
		for b := 0; b < 4; b++ {
			for c := 0; c < 4; c++ {
				for d := 0; d < 4; d++ {
					var s float64
					for lam := 0; lam < 4; lam++ {
						s += g[a][lam] * R[lam][b][c][d]
					}
					out[a][b][c][d] = s
				}
			}
		}
	}
	return out
}

// Ricci returns R_σν = R^ρ_σρν.
func Ricci(R Tensor4) Mat {
	var ric Mat
	for sigma := 0; sigma < 4; sigma++ {
		for nu := 0; nu < 4; nu++ {
			var s float64
			for rho := 0; rho < 4; rho++ {
				s += R[rho][sigma][rho][nu]
			}
			ric[sigma][nu] = s
		}
	}
	return ric
}

// ScalarCurvature returns R = g^μν R_μν.
func ScalarCurvature(ginv, ric Mat) float64 {
	var s float64
	for mu := 0; mu < 4; mu++ {
		for nu := 0; nu < 4; nu++ {
			s += ginv[mu][nu] * ric[mu][nu]
		}
	}
	return s
}

// Einstein returns G_μν = R_μν − ½ g_μν R.
func Einstein(g, ric Mat, scalar float64) Mat {
	var G Mat
	for mu := 0; mu < 4; mu++ {
		for nu := 0; nu < 4; nu++ {
			G[mu][nu] = ric[mu][nu] - 0.5*scalar*g[mu][nu]
		}
	}
	return G
}

// KretschmannScalar returns R_μνρσ R^μνρσ.
func KretschmannScalar(g, ginv Mat, R Tensor4) float64 {
	down := LowerRiemann(g, R)
	var up Tensor4
	// Raise all four indices: R^αβγδ = g^αρ g^βσ g^γμ g^δν R_ρσμν.
	for a := 0; a < 4; a++ {
		for b := 0; b < 4; b++ {
			for c := 0; c < 4; c++ {
				for d := 0; d < 4; d++ {
					var s float64
					for p := 0; p < 4; p++ {
						gap := ginv[a][p]
						if gap == 0 {
							continue
						}
						for q := 0; q < 4; q++ {
							gbq := ginv[b][q]
							if gbq == 0 {
								continue
							}
							gpq := gap * gbq
							for u := 0; u < 4; u++ {
								gcu := ginv[c][u]
								if gcu == 0 {
									continue
								}
								for v := 0; v < 4; v++ {
									s += gpq * gcu * ginv[d][v] * down[p][q][u][v]
								}
							}
						}
					}
					up[a][b][c][d] = s
				}
			}
		}
	}
	var K float64
	for a := 0; a < 4; a++ {
		for b := 0; b < 4; b++ {
			for c := 0; c < 4; c++ {
				for d := 0; d < 4; d++ {
					K += down[a][b][c][d] * up[a][b][c][d]
				}
			}
		}
	}
	return K
}

// Curvature is a pointwise curvature report.
type Curvature struct {
	Kretschmann float64
	Scalar      float64
	Ricci       Mat
	Einstein    Mat
	Riemann     Tensor4
}

// CurvatureAt computes the curvature tensors at x.
func CurvatureAt(m Metric, x Vec, h float64) (Curvature, bool) {
	R, ok := Riemann(m, x, h)
	if !ok {
		return Curvature{}, false
	}
	g := m.Cov(x)
	ginv, ok := Inverse(m, x)
	if !ok {
		return Curvature{}, false
	}
	ric := Ricci(R)
	scalar := ScalarCurvature(ginv, ric)
	return Curvature{
		Kretschmann: KretschmannScalar(g, ginv, R),
		Scalar:      scalar,
		Ricci:       ric,
		Einstein:    Einstein(g, ric, scalar),
		Riemann:     R,
	}, true
}

// Component is one named tensor component.
type Component struct {
	Name  string
	Value float64
}

// ChristoffelComponents lists components whose absolute value exceeds tol.
func ChristoffelComponents(m Metric, x Vec, tol float64) ([]Component, bool) {
	g, ok := Christoffel(m, x)
	if !ok {
		return nil, false
	}
	names := m.CoordNames()
	var out []Component
	for rho := 0; rho < 4; rho++ {
		for mu := 0; mu < 4; mu++ {
			for nu := mu; nu < 4; nu++ {
				v := g[rho][mu][nu]
				if math.Abs(v) <= tol {
					continue
				}
				out = append(out, Component{
					Name:  fmt.Sprintf("Γ^%s_%s%s", names[rho], names[mu], names[nu]),
					Value: v,
				})
			}
		}
	}
	return out, true
}
