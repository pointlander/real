// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

// Minkowski is flat spacetime in Cartesian coordinates (t, x, y, z).
type Minkowski struct{}

func (Minkowski) Name() string { return "Minkowski" }

func (Minkowski) CoordNames() [4]string { return [4]string{"t", "x", "y", "z"} }

func (Minkowski) Cov(Vec) Mat {
	return Mat{
		{-1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}
}

func (Minkowski) Con(Vec) (Mat, bool) {
	return Mat{
		{-1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}, true
}

func (Minkowski) DCov(Vec) Tensor3 { return Tensor3{} }
