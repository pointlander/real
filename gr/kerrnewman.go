// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import "math"

// KerrNewman is the Kerr–Newman black hole in Boyer–Lindquist coordinates
// (t, r, θ, φ). A is the angular momentum per unit mass and Q is the charge.
// A = Q = 0 is Schwarzschild. Q = 0 is Kerr. A = 0 is Reissner–Nordström.
type KerrNewman struct {
	M, A, Q float64
}

// Schwarzschild returns a non-spinning, uncharged black hole of mass M.
func Schwarzschild(M float64) KerrNewman { return KerrNewman{M: M} }

// Kerr returns an uncharged rotating black hole.
func Kerr(M, A float64) KerrNewman { return KerrNewman{M: M, A: A} }

// ReissnerNordstrom returns a charged, non-spinning black hole.
func ReissnerNordstrom(M, Q float64) KerrNewman { return KerrNewman{M: M, Q: Q} }

func (k KerrNewman) Name() string {
	switch {
	case k.A == 0 && k.Q == 0:
		return "Schwarzschild"
	case k.A == 0:
		return "Reissner-Nordström"
	case k.Q == 0:
		return "Kerr"
	default:
		return "Kerr-Newman"
	}
}

func (k KerrNewman) CoordNames() [4]string {
	return [4]string{"t", "r", "theta", "phi"}
}

func (k KerrNewman) sigmaDelta(r, theta float64) (sig, delta, rho2, s, c float64) {
	s = math.Sin(theta)
	c = math.Cos(theta)
	a2 := k.A * k.A
	sig = r*r + a2*c*c
	delta = r*r - 2*k.M*r + a2 + k.Q*k.Q
	rho2 = 2*k.M*r - k.Q*k.Q
	return sig, delta, rho2, s, c
}

// Cov returns g_μν.
func (k KerrNewman) Cov(x Vec) Mat {
	r, th := x[1], x[2]
	sig, delta, rho2, s, _ := k.sigmaDelta(r, th)
	s2 := s * s
	a2 := k.A * k.A
	r2 := r * r
	var g Mat
	g[0][0] = -1 + rho2/sig
	g[1][1] = sig / delta
	g[2][2] = sig
	g[0][3] = -k.A * s2 * rho2 / sig
	g[3][0] = g[0][3]
	rp2 := r2 + a2
	g[3][3] = s2 * (rp2*rp2 - delta*a2*s2) / sig
	return g
}

// DCov returns ∂_λ g_μν.
func (k KerrNewman) DCov(x Vec) Tensor3 {
	r, th := x[1], x[2]
	a := k.A
	M := k.M
	s := math.Sin(th)
	c := math.Cos(th)
	s2 := s * s
	sc := s * c
	a2 := a * a
	r2 := r * r
	sig := r2 + a2*c*c
	sig2 := sig * sig
	delta := r2 - 2*M*r + a2 + k.Q*k.Q
	rho2 := 2*M*r - k.Q*k.Q

	sigR := 2 * r
	sigTh := -2 * a2 * sc
	deltaR := 2*r - 2*M
	rho2R := 2 * M

	var dg Tensor3
	dg[1][0][0] = (rho2R*sig - rho2*sigR) / sig2
	dg[2][0][0] = -rho2 * sigTh / sig2

	dlt2 := delta * delta
	dg[1][1][1] = (sigR*delta - sig*deltaR) / dlt2
	dg[2][1][1] = sigTh / delta

	dg[1][2][2] = sigR
	dg[2][2][2] = sigTh

	dRhoR := (rho2R*sig - rho2*sigR) / sig2
	dg[1][0][3] = -a * s2 * dRhoR
	dg[1][3][0] = dg[1][0][3]
	dS2RhoTh := rho2 * (2*sc*sig - s2*sigTh) / sig2
	dg[2][0][3] = -a * dS2RhoTh
	dg[2][3][0] = dg[2][0][3]

	rp2 := r2 + a2
	P := rp2*rp2 - delta*a2*s2
	dPr := 4*r*rp2 - deltaR*a2*s2
	dPth := -delta * a2 * 2 * sc
	dg[1][3][3] = s2 * (dPr*sig - P*sigR) / sig2
	dg[2][3][3] = (2*sc*P*sig + s2*dPth*sig - s2*P*sigTh) / sig2
	return dg
}

// Horizons returns the outer and inner horizons. ok is false for a naked singularity.
func (k KerrNewman) Horizons() (outer, inner float64, ok bool) {
	disc := k.M*k.M - k.A*k.A - k.Q*k.Q
	if disc < 0 || k.M <= 0 {
		return 0, 0, false
	}
	s := math.Sqrt(disc)
	return k.M + s, k.M - s, true
}

// ErgosphereRadius is the outer static-limit radius at polar angle theta.
func (k KerrNewman) ErgosphereRadius(theta float64) (float64, bool) {
	c := math.Cos(theta)
	disc := k.M*k.M - k.A*k.A*c*c - k.Q*k.Q
	if disc < 0 {
		return 0, false
	}
	return k.M + math.Sqrt(disc), true
}

// KretschmannExact is the closed-form Kretschmann scalar of the Kerr–Newman
// geometry (Henry 1999; Kraniotis 2022).
func (k KerrNewman) KretschmannExact(r, theta float64) float64 {
	a := k.A
	m := k.M
	q := k.Q
	c2 := math.Cos(theta)
	c2 *= c2
	c4 := c2 * c2
	c6 := c4 * c2
	a2 := a * a
	a4 := a2 * a2
	a6 := a4 * a2
	q2 := q * q
	q4 := q2 * q2
	r2 := r * r
	r4 := r2 * r2
	r5 := r4 * r
	r6 := r4 * r2
	sig := r2 + a2*c2
	term := -48*c6*a6*m*m +
		720*(m*m*r2-(2.0/3.0)*m*q2*r+(7.0/90.0)*q4)*a4*c4 -
		720*(m*m*r2-(4.0/3.0)*m*q2*r+(17.0/45.0)*q4)*a2*r2*c2 +
		48*m*m*r6 - 96*m*q2*r5 + 56*q4*r4
	return term / math.Pow(sig, 6)
}

// SchwarzschildKretschmann is 48 M² / r⁶.
func SchwarzschildKretschmann(M, r float64) float64 {
	return 48 * M * M / math.Pow(r, 6)
}

// KerrISCORadius returns the equatorial innermost stable circular orbit.
// The upper sign in the Bardeen–Press–Teukolsky formula is the prograde orbit.
func KerrISCORadius(M, a float64, prograde bool) (float64, bool) {
	if M <= 0 || math.Abs(a) > M {
		return 0, false
	}
	chi := a / M
	chi2 := chi * chi
	z1 := 1 + math.Cbrt(1-chi2)*(math.Cbrt(1+chi)+math.Cbrt(1-chi))
	z2 := math.Sqrt(3*chi2 + z1*z1)
	inner := (3 - z1) * (3 + z1 + 2*z2)
	if inner < 0 {
		inner = 0
	}
	rad := math.Sqrt(inner)
	r := 3 + z2
	if prograde {
		r -= rad
	} else {
		r += rad
	}
	return r * M, true
}
