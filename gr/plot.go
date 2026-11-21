// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import (
	"math"
	"strings"
)

// PlotEquatorial draws r–φ samples on a text grid.
// samples are coordinate tuples (t, r, θ, φ). horizon, if positive, is drawn
// as a circle. The origin is the coordinate center.
func PlotEquatorial(samples []OrbitSample, horizon float64, width, height int) string {
	if width < 8 {
		width = 8
	}
	if height < 6 {
		height = 6
	}
	maxR := horizon
	for _, s := range samples {
		rr := s.X[1]
		if rr > maxR {
			maxR = rr
		}
	}
	if maxR <= 0 {
		maxR = 1
	}
	maxR *= 1.05
	cell := make([][]rune, height)
	for i := range cell {
		cell[i] = make([]rune, width)
		for j := range cell[i] {
			cell[i][j] = ' '
		}
	}
	xy := func(x, y float64) (int, int, bool) {
		col := int(math.Round((x/maxR + 1) * 0.5 * float64(width-1)))
		row := int(math.Round((1 - y/maxR) * 0.5 * float64(height-1)))
		if col < 0 || row < 0 || col >= width || row >= height {
			return 0, 0, false
		}
		return col, row, true
	}
	put := func(x, y float64, ch rune) {
		c, r, ok := xy(x, y)
		if !ok {
			return
		}
		if cell[r][c] == '*' && ch != '*' {
			return
		}
		cell[r][c] = ch
	}
	if horizon > 0 {
		for i := 0; i < 240; i++ {
			a := 2 * math.Pi * float64(i) / 240
			put(horizon*math.Cos(a), horizon*math.Sin(a), '·')
		}
	}
	for _, s := range samples {
		r := s.X[1]
		phi := s.X[3]
		put(r*math.Cos(phi), r*math.Sin(phi), '*')
	}
	put(0, 0, '+')
	var b strings.Builder
	for i := 0; i < height; i++ {
		b.WriteString(string(cell[i]))
		if i+1 < height {
			b.WriteByte('\n')
		}
	}
	return b.String()
}
