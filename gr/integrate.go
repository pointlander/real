// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import "math"

// dopri5 integrates y' = f(t, y) from t0 to t1 with the Dormand–Prince 5(4) pair.
// f appends the derivative into dst and returns false if the right-hand side fails.
// stop, if non-nil, ends the integration successfully after an accepted step.
// observe, if non-nil, is called at t0 and after every accepted step.
func dopri5(
	f func(t float64, y, dst []float64) bool,
	t0 float64,
	y0 []float64,
	t1, atol, rtol, h0, hmin, hmax float64,
	maxSteps int,
	stop func(t float64, y []float64) bool,
	observe func(t float64, y []float64),
) (tEnd float64, yEnd []float64, steps, rejects int, stopped bool, ok bool) {
	n := len(y0)
	y := append([]float64(nil), y0...)
	ynew := make([]float64, n)
	ks := make([][]float64, 7)
	for i := range ks {
		ks[i] = make([]float64, n)
	}
	stage := make([]float64, n)
	t := t0
	h := h0
	if h == 0 {
		h = 1e-2
	}
	if t1 < t0 {
		if h > 0 {
			h = -h
		}
	} else if h < 0 {
		h = -h
	}
	if observe != nil {
		observe(t, y)
	}
	if !f(t, y, ks[0]) {
		return t, y, 0, 0, false, false
	}
	if maxSteps <= 0 {
		maxSteps = 1_000_000
	}
	for steps < maxSteps {
		if stop != nil && stop(t, y) {
			return t, y, steps, rejects, true, true
		}
		remain := t1 - t
		if remain == 0 {
			return t, y, steps, rejects, false, true
		}
		if math.Abs(h) > math.Abs(remain) {
			h = remain
		}
		if math.Abs(h) < hmin {
			return t, y, steps, rejects, false, false
		}
		accepted := false
		for tries := 0; tries < 20; tries++ {
			if !dopriStages(f, t, h, y, ks, stage) {
				return t, y, steps, rejects, false, false
			}
			for i := 0; i < n; i++ {
				s := y[i]
				for j := 0; j < 7; j++ {
					s += h * dpB5[j] * ks[j][i]
				}
				ynew[i] = s
			}
			var err2 float64
			for i := 0; i < n; i++ {
				var ei float64
				for j := 0; j < 7; j++ {
					ei += h * (dpB5[j] - dpB4[j]) * ks[j][i]
				}
				sc := atol + rtol*math.Max(math.Abs(y[i]), math.Abs(ynew[i]))
				r := ei / sc
				err2 += r * r
			}
			err := math.Sqrt(err2 / float64(n))
			if err == 0 || math.IsNaN(err) {
				err = 1e-16
			}
			factor := 0.9 * math.Pow(err, -0.2)
			if factor < 0.2 {
				factor = 0.2
			} else if factor > 5 {
				factor = 5
			}
			if err <= 1 {
				t += h
				copy(y, ynew)
				steps++
				if observe != nil {
					observe(t, y)
				}
				h *= factor
				if math.Abs(h) > hmax && hmax > 0 {
					if h > 0 {
						h = hmax
					} else {
						h = -hmax
					}
				}
				// FSAL: the last stage was not evaluated at ynew.
				if !f(t, y, ks[0]) {
					return t, y, steps, rejects, false, false
				}
				accepted = true
				break
			}
			rejects++
			h *= factor
			if math.Abs(h) < hmin {
				return t, y, steps, rejects, false, false
			}
		}
		if !accepted {
			return t, y, steps, rejects, false, false
		}
	}
	return t, y, steps, rejects, false, false
}

// Dormand–Prince coefficients. Stage 0 is unused in the a-table.
var dpA = [7][6]float64{
	{},
	{1.0 / 5.0},
	{3.0 / 40.0, 9.0 / 40.0},
	{44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0},
	{19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0},
	{9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0},
	{35.0 / 384.0, 0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0},
}

var dpC = [7]float64{0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1, 1}

var dpB5 = [7]float64{35.0 / 384.0, 0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0}

var dpB4 = [7]float64{5179.0 / 57600.0, 0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0}

func dopriStages(f func(t float64, y, dst []float64) bool, t, h float64, y []float64, ks [][]float64, stage []float64) bool {
	n := len(y)
	for s := 1; s < 7; s++ {
		for i := 0; i < n; i++ {
			sum := y[i]
			for j := 0; j < s; j++ {
				sum += h * dpA[s][j] * ks[j][i]
			}
			stage[i] = sum
		}
		if !f(t+dpC[s]*h, stage, ks[s]) {
			return false
		}
	}
	return true
}
