// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import (
	"fmt"
	"math"
)

// XOR orbit: a stable timelike circle, far from the photon sphere.
// A full turn at this radius closes to about 1e-6, so half a turn,
// a full turn, and no turn are easy to tell apart.
const (
	xorMass   = 1.0
	xorRadius = 10.0
)

// XORTrace is one evaluation of the geodesic XOR.
type XORTrace struct {
	Bit bool    // true when the final azimuth is nearer π than 2πℤ
	Phi float64 // final coordinate azimuth
	R   float64 // final radius
}

// XOR reports a XOR b by evolving one timelike circular geodesic in the
// Schwarzschild geometry. Each true input advances the azimuth by half a
// turn; a false input is the identity (a geodesic of zero length). The
// output bit is read from the final azimuth: one half-turn ends near π,
// and two half-turns end near 2π, which is the same place as none.
func XOR(a, b bool) (XORTrace, error) {
	m := Schwarzschild(xorMass)
	pos, _, ok, _ := CircularOrbits(m, xorRadius, math.Pi/2)
	if !ok {
		return XORTrace{}, fmt.Errorf("no timelike circular orbit at r=%g", xorRadius)
	}
	st := State{X: pos.X, V: pos.V}
	for _, bit := range []bool{a, b} {
		delta := 0.0
		if bit {
			delta = math.Pi
		}
		next, err := advanceAzimuth(m, st, delta)
		if err != nil {
			return XORTrace{}, err
		}
		st = next
	}
	return XORTrace{Bit: nearerOddHalf(st.X[3]), Phi: st.X[3], R: st.X[1]}, nil
}

// advanceAzimuth integrates the geodesic until φ has grown by delta.
// delta = 0 stops on the initial point, which is the identity.
func advanceAzimuth(m Metric, st State, delta float64) (State, error) {
	if st.V[3] <= 0 {
		return State{}, fmt.Errorf("need a prograde orbit, got u^φ = %g", st.V[3])
	}
	target := st.X[3] + delta
	cfg := DefaultIntegrate()
	cfg.Kind = Timelike
	cfg.Project = true
	cfg.Hmax = 0.25
	cfg.Rtol = 1e-10
	// Two full turns of proper time is more than a half-turn needs.
	cfg.T1 = 4 * math.Pi / st.V[3]
	cfg.Stop = func(s State) bool {
		return s.X[3] >= target
	}
	res := IntegrateGeodesic(m, st.X, st.V, cfg)
	if !res.Finished || !res.Stopped {
		return State{}, fmt.Errorf("geodesic stopped short of Δφ = %g (φ = %g)", delta, res.State.X[3])
	}
	return res.State, nil
}

// nearerOddHalf is true when phi, modulo 2π, lies in (π/2, 3π/2).
func nearerOddHalf(phi float64) bool {
	turn := math.Mod(phi, 2*math.Pi)
	if turn < 0 {
		turn += 2 * math.Pi
	}
	return math.Abs(turn-math.Pi) < math.Pi/2
}
