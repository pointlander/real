// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import (
	"fmt"
	"math"
)

// TwoQubit is a pure state in the computational basis |control, target⟩.
// Element 2*control+target holds that basis amplitude.
type TwoQubit [4]complex128

// CNOTTrace is one application of the geodesic CNOT.
type CNOTTrace struct {
	Out TwoQubit
	Phi [4]float64 // final azimuth of each input basis geodesic
	R   [4]float64 // final radius of each input basis geodesic
}

// CNOT applies the quantum controlled-NOT. Each computational-basis
// component flies the same timelike circle used by XOR. The target bit
// is the starting azimuth, 0 or π. A control value of 1 advances that
// geodesic by half a turn, which flips the azimuth parity; a control
// value of 0 leaves it on the initial point. The amplitude is carried
// unchanged onto the destination labeled by the final azimuth, so a
// superposition stays coherent. On |+⟩|0⟩ the two ships land on |00⟩
// and |11⟩.
func CNOT(in TwoQubit) (CNOTTrace, error) {
	m := Schwarzschild(xorMass)
	pos, _, ok, _ := CircularOrbits(m, xorRadius, math.Pi/2)
	if !ok {
		return CNOTTrace{}, fmt.Errorf("no timelike circular orbit at r=%g", xorRadius)
	}
	var tr CNOTTrace
	for i := 0; i < 4; i++ {
		control := i >> 1
		target := i & 1
		st := State{X: pos.X, V: pos.V}
		st.X[3] = float64(target) * math.Pi
		delta := 0.0
		if control == 1 {
			delta = math.Pi
		}
		next, err := advanceAzimuth(m, st, delta)
		if err != nil {
			return CNOTTrace{}, err
		}
		dest := control << 1
		if nearerOddHalf(next.X[3]) {
			dest |= 1
		}
		tr.Out[dest] += in[i]
		tr.Phi[i] = next.X[3]
		tr.R[i] = next.X[1]
	}
	return tr, nil
}
