// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import "math"

// Vec is a 4-vector, v^μ or a coordinate tuple x^μ.
type Vec [4]float64

// Mat is a rank-2 tensor with both indices in the same position, T_μν or T^μν.
type Mat [4][4]float64

// Tensor3 is T^ρ_μν or ∂_λ g_μν, stored as [ρ][μ][ν] or [λ][μ][ν].
type Tensor3 [4][4][4]float64

// Tensor4 is a rank-4 tensor stored as [a][b][c][d].
type Tensor4 [4][4][4][4]float64

// Contract returns g_μν u^μ v^ν.
func Contract(g Mat, u, v Vec) float64 {
	var s float64
	for mu := 0; mu < 4; mu++ {
		for nu := 0; nu < 4; nu++ {
			s += g[mu][nu] * u[mu] * v[nu]
		}
	}
	return s
}

// MulMat returns a*b.
func MulMat(a, b Mat) Mat {
	var c Mat
	for i := 0; i < 4; i++ {
		for k := 0; k < 4; k++ {
			aik := a[i][k]
			if aik == 0 {
				continue
			}
			for j := 0; j < 4; j++ {
				c[i][j] += aik * b[k][j]
			}
		}
	}
	return c
}

// Inv inverts a 4×4 matrix. ok is false when a pivot is numerically zero.
func (m Mat) Inv() (Mat, bool) {
	var a [4][8]float64
	scale := 0.0
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			a[i][j] = m[i][j]
			if v := math.Abs(m[i][j]); v > scale {
				scale = v
			}
		}
		a[i][i+4] = 1
	}
	if scale == 0 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		return Mat{}, false
	}
	tol := 1e-14 * scale
	for col := 0; col < 4; col++ {
		piv := col
		max := math.Abs(a[col][col])
		for r := col + 1; r < 4; r++ {
			if v := math.Abs(a[r][col]); v > max {
				max = v
				piv = r
			}
		}
		if max < tol {
			return Mat{}, false
		}
		if piv != col {
			a[col], a[piv] = a[piv], a[col]
		}
		p := a[col][col]
		for j := col; j < 8; j++ {
			a[col][j] /= p
		}
		for r := 0; r < 4; r++ {
			if r == col {
				continue
			}
			f := a[r][col]
			if f == 0 {
				continue
			}
			for j := col; j < 8; j++ {
				a[r][j] -= f * a[col][j]
			}
		}
	}
	var inv Mat
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			inv[i][j] = a[i][j+4]
		}
	}
	return inv, true
}

// MaxAbs returns the largest absolute entry.
func (m Mat) MaxAbs() float64 {
	max := 0.0
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if v := math.Abs(m[i][j]); v > max {
				max = v
			}
		}
	}
	return max
}

// MaxAbs3 returns the largest absolute entry of a rank-3 tensor.
func MaxAbs3(t Tensor3) float64 {
	max := 0.0
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 4; k++ {
				if v := math.Abs(t[i][j][k]); v > max {
					max = v
				}
			}
		}
	}
	return max
}

// MaxAbs4 returns the largest absolute entry of a rank-4 tensor.
func MaxAbs4(t Tensor4) float64 {
	max := 0.0
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 4; k++ {
				for l := 0; l < 4; l++ {
					if v := math.Abs(t[i][j][k][l]); v > max {
						max = v
					}
				}
			}
		}
	}
	return max
}
