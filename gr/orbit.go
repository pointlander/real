// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gr

import "math"

// Circular is an equatorial circular orbit in a chart (t, r, θ, φ).
type Circular struct {
	X, V       Vec
	Omega      float64 // dφ/dt
	E, L       float64
	NormFactor float64 // g_tt + 2 Ω g_tφ + Ω² g_φφ; negative for timelike
}

// CircularOrbits returns the two equatorial angular velocities that keep r̈ = 0.
// A root is reported only when the tangent can be normalized to a future-directed
// timelike vector (norm factor < 0). positiveL selects L > 0; the other return
// value is the opposite branch. Either ok flag may be false.
func CircularOrbits(m Metric, r, theta float64) (pos, neg Circular, posOK, negOK bool) {
	x := Vec{0, r, theta, 0}
	gamma, ok := Christoffel(m, x)
	if !ok {
		return Circular{}, Circular{}, false, false
	}
	o1, o2, ok := circularOmegas(gamma)
	if !ok {
		return Circular{}, Circular{}, false, false
	}
	c1, ok1 := circularFromOmega(m, x, o1)
	c2, ok2 := circularFromOmega(m, x, o2)
	if ok1 {
		if c1.L >= 0 {
			pos, posOK = c1, true
		} else {
			neg, negOK = c1, true
		}
	}
	if ok2 {
		if c2.L >= 0 {
			pos, posOK = c2, true
		} else {
			neg, negOK = c2, true
		}
	}
	return pos, neg, posOK, negOK
}

func circularOmegas(gamma Tensor3) (o1, o2 float64, ok bool) {
	A := gamma[1][3][3]
	B := 2 * gamma[1][0][3]
	C := gamma[1][0][0]
	disc := B*B - 4*A*C
	if disc < 0 {
		return 0, 0, false
	}
	if math.Abs(A) < 1e-14*math.Max(1, math.Abs(B)+math.Abs(C)) {
		if math.Abs(B) < 1e-18 {
			return 0, 0, false
		}
		o := -C / B
		return o, o, true
	}
	s := math.Sqrt(disc)
	return (-B + s) / (2 * A), (-B - s) / (2 * A), true
}

func circularFromOmega(m Metric, x Vec, omega float64) (Circular, bool) {
	g := m.Cov(x)
	nf := g[0][0] + 2*omega*g[0][3] + omega*omega*g[3][3]
	if nf >= -1e-12 {
		return Circular{}, false
	}
	ut := 1 / math.Sqrt(-nf)
	v := Vec{ut, 0, 0, omega * ut}
	E, L := ConstantsOfMotion(m, x, v)
	if E <= 0 {
		ut = -ut
		v[0] = ut
		v[3] = omega * ut
		E, L = ConstantsOfMotion(m, x, v)
	}
	return Circular{X: x, V: v, Omega: omega, E: E, L: L, NormFactor: nf}, true
}

// PhotonOrbit finds the equatorial photon orbit on one L branch by bisection.
// rLo should lie inside the photon orbit (no timelike circle) and rHi outside it.
func PhotonOrbit(m Metric, rLo, rHi float64, positiveL bool) (float64, bool) {
	f := func(r float64) float64 {
		nf, ok := branchNorm(m, r, positiveL)
		if !ok {
			return math.NaN()
		}
		return nf
	}
	return bisectSign(f, rLo, rHi, 1e-10)
}

// branchNorm is g_μν k^μ k^ν for k = (1, 0, 0, Ω) on the chosen circular branch.
func branchNorm(m Metric, r float64, positiveL bool) (float64, bool) {
	x := Vec{0, r, math.Pi / 2, 0}
	gamma, ok := Christoffel(m, x)
	if !ok {
		return 0, false
	}
	o1, o2, ok := circularOmegas(gamma)
	if !ok {
		return 0, false
	}
	g := m.Cov(x)
	n1 := g[0][0] + 2*o1*g[0][3] + o1*o1*g[3][3]
	n2 := g[0][0] + 2*o2*g[0][3] + o2*o2*g[3][3]
	_, L1 := constantsFromOmega(g, o1)
	_, L2 := constantsFromOmega(g, o2)
	if positiveL {
		if L1 >= L2 {
			return n1, true
		}
		return n2, true
	}
	if L1 < L2 {
		return n1, true
	}
	return n2, true
}

func constantsFromOmega(g Mat, omega float64) (E, L float64) {
	// Affine choice k^t = 1, k^φ = Ω. Valid for the ratio L/E even when null.
	ut := g[0][0] + g[0][3]*omega
	up := g[3][0] + g[3][3]*omega
	return -ut, up
}

// CriticalImpact returns b = L/E of the equatorial photon orbit.
func CriticalImpact(m Metric, rPhoton float64, positiveL bool) (float64, bool) {
	x := Vec{0, rPhoton, math.Pi / 2, 0}
	gamma, ok := Christoffel(m, x)
	if !ok {
		return 0, false
	}
	o1, o2, ok := circularOmegas(gamma)
	if !ok {
		return 0, false
	}
	g := m.Cov(x)
	e1, l1 := constantsFromOmega(g, o1)
	e2, l2 := constantsFromOmega(g, o2)
	var E, L float64
	if positiveL == (l1 >= l2) {
		E, L = e1, l1
	} else {
		E, L = e2, l2
	}
	if math.Abs(E) < 1e-14 {
		return 0, false
	}
	return L / E, true
}

// ISCO minimizes the circular-orbit energy on (rLo, rHi).
func ISCO(m Metric, rLo, rHi float64, positiveL bool) (float64, bool) {
	f := func(r float64) float64 {
		c, ok := circularBranch(m, r, positiveL)
		if !ok {
			return math.NaN()
		}
		return c.E
	}
	// Shrink the bracket onto the interval where circular orbits exist.
	lo, hi, ok := finiteBracket(f, rLo, rHi)
	if !ok {
		return 0, false
	}
	r, _ := goldenMin(f, lo, hi, 1e-8*(1+hi))
	if math.IsNaN(f(r)) {
		return 0, false
	}
	return r, true
}

func circularBranch(m Metric, r float64, positiveL bool) (Circular, bool) {
	pos, neg, pok, nok := CircularOrbits(m, r, math.Pi/2)
	if positiveL {
		return pos, pok
	}
	return neg, nok
}

func finiteBracket(f func(float64) float64, lo, hi float64) (float64, float64, bool) {
	const n = 80
	prevOK := false
	left, right := lo, lo
	found := false
	for i := 0; i <= n; i++ {
		r := lo + (hi-lo)*float64(i)/float64(n)
		v := f(r)
		ok := !math.IsNaN(v) && !math.IsInf(v, 0)
		if ok && !prevOK {
			left = r
			found = true
		}
		if ok {
			right = r
		}
		prevOK = ok
	}
	if !found || right <= left {
		return 0, 0, false
	}
	return left, right, true
}

func goldenMin(f func(float64) float64, a, b, tol float64) (float64, float64) {
	const invPhi = 0.6180339887498949
	c := b - invPhi*(b-a)
	d := a + invPhi*(b-a)
	fc, fd := f(c), f(d)
	for i := 0; i < 80 && b-a > tol; i++ {
		if fc < fd {
			b, d, fd = d, c, fc
			c = b - invPhi*(b-a)
			fc = f(c)
		} else {
			a, c, fc = c, d, fd
			d = a + invPhi*(b-a)
			fd = f(d)
		}
	}
	if fc < fd {
		return c, fc
	}
	return d, fd
}

func bisectSign(f func(float64) float64, a, b, tol float64) (float64, bool) {
	fa, fb := f(a), f(b)
	if math.IsNaN(fa) || math.IsNaN(fb) || fa*fb > 0 {
		return 0, false
	}
	for i := 0; i < 80 && math.Abs(b-a) > tol; i++ {
		m := 0.5 * (a + b)
		fm := f(m)
		if math.IsNaN(fm) {
			return 0, false
		}
		if fa*fm <= 0 {
			b, fb = m, fm
		} else {
			a, fa = m, fm
		}
	}
	return 0.5 * (a + b), true
}

// TurningPointConstants finds E > 0 and L for an equatorial orbit that turns
// at r1 and r2. The metric must have g_tφ = 0 (static), in coordinates (t, r, θ, φ).
func TurningPointConstants(m Metric, r1, r2, theta float64) (E, L float64, ok bool) {
	A1, C1, D1, ok1 := staticTL(m, r1, theta)
	A2, C2, D2, ok2 := staticTL(m, r2, theta)
	if !ok1 || !ok2 {
		return 0, 0, false
	}
	// C E² + A L² = -D, for g(u, u) = -1 and u^r = 0.
	// [C1 A1; C2 A2] [E²; L²] = [-D1; -D2]
	det := C1*A2 - C2*A1
	if math.Abs(det) < 1e-18 {
		return 0, 0, false
	}
	e2 := (-D1*A2 - A1*(-D2)) / det
	l2 := (C1*(-D2) - (-D1)*C2) / det
	if e2 <= 0 || l2 <= 0 {
		return 0, 0, false
	}
	return math.Sqrt(e2), math.Sqrt(l2), true
}

func staticTL(m Metric, r, theta float64) (gtt, gpp, det float64, ok bool) {
	g := m.Cov(Vec{0, r, theta, 0})
	if math.Abs(g[0][3]) > 1e-8*math.Max(1, math.Abs(g[0][0])) {
		return 0, 0, 0, false
	}
	gtt = g[0][0]
	gpp = g[3][3]
	det = gtt*gpp - g[0][3]*g[0][3]
	if gtt >= 0 || gpp <= 0 {
		return 0, 0, 0, false
	}
	return gtt, gpp, det, true
}

// OrbitSample is one point on an integrated orbit.
type OrbitSample struct {
	Lambda float64
	X, V   Vec
}

// OrbitResult is a bound equatorial orbit started at periapsis.
type OrbitResult struct {
	E, L       float64
	Samples    []OrbitSample
	PeriLambda []float64
	PeriPhi    []float64
	PeriR      []float64
	// Precession is the mean φ advance between periapses, minus 2π.
	Precession float64
	Finished   bool
}

// BoundOrbit integrates a timelike equatorial orbit with turning points rp < ra.
// The metric must be static (g_tφ = 0). periods is the number of radial periods.
func BoundOrbit(m Metric, rp, ra float64, periods int, cfg IntegrateConfig) OrbitResult {
	var out OrbitResult
	if periods < 1 {
		periods = 1
	}
	E, L, ok := TurningPointConstants(m, rp, ra, math.Pi/2)
	if !ok {
		return out
	}
	out.E, out.L = E, L
	x := Vec{0, rp, math.Pi / 2, 0}
	v, ok := VelocityFromConserved(m, x, E, L, Timelike, false)
	if !ok {
		return out
	}
	// Leave periapsis outward.
	if v[1] < 0 {
		v[1] = -v[1]
	}
	var prev OrbitSample
	have := false
	seenApo := false
	user := cfg.Observe
	cfg.Kind = Timelike
	cfg.Observe = func(s State) {
		sm := OrbitSample{Lambda: s.Lambda, X: s.X, V: s.V}
		out.Samples = append(out.Samples, sm)
		if user != nil {
			user(s)
		}
		if have {
			if prev.V[1] > 0 && s.V[1] < 0 {
				seenApo = true
			}
			if seenApo && prev.V[1] < 0 && s.V[1] >= 0 {
				frac := 0.0
				den := s.V[1] - prev.V[1]
				if den != 0 {
					frac = -prev.V[1] / den
				}
				if frac < 0 {
					frac = 0
				} else if frac > 1 {
					frac = 1
				}
				out.PeriLambda = append(out.PeriLambda, prev.Lambda+frac*(s.Lambda-prev.Lambda))
				out.PeriPhi = append(out.PeriPhi, prev.X[3]+frac*(s.X[3]-prev.X[3]))
				out.PeriR = append(out.PeriR, prev.X[1]+frac*(s.X[1]-prev.X[1]))
				seenApo = false
			}
		}
		prev = sm
		have = true
	}
	cfg.Stop = func(State) bool {
		return len(out.PeriPhi) >= periods
	}
	res := IntegrateGeodesic(m, x, v, cfg)
	out.Finished = res.Finished
	if len(out.PeriPhi) >= 1 {
		// φ at the initial periapsis is 0.
		var sum float64
		prev := 0.0
		n := 0
		for _, phi := range out.PeriPhi {
			sum += (phi - prev) - 2*math.Pi
			prev = phi
			n++
		}
		out.Precession = sum / float64(n)
	}
	return out
}

// SchwarzschildPrecession is the exact coordinate precession per radial period,
// 2 ∫_rp^ra dφ − 2π, for a static Schwarzschild orbit with the given constants.
func SchwarzschildPrecession(M, E, L, rp, ra float64) float64 {
	dphi := schwarzschildPhiIntegral(M, E, L, rp, ra, 4000)
	return 2*dphi - 2*math.Pi
}

func schwarzschildPhiIntegral(M, E, L, rp, ra float64, n int) float64 {
	if n < 16 {
		n = 16
	}
	half := 0.5 * (ra - rp)
	mid := 0.5 * (ra + rp)
	h := math.Pi / float64(n)
	var s float64
	for i := 0; i < n; i++ {
		chi := (float64(i) + 0.5) * h
		r := mid + half*math.Cos(chi)
		dr := half * math.Sin(chi)
		V := (1 - 2*M/r) * (1 + L*L/(r*r))
		rad := E*E - V
		if rad < 1e-18 {
			rad = 1e-18
		}
		s += (L / (r * r)) / math.Sqrt(rad) * dr
	}
	return s * h
}

// Deflection is a null ray sent inward from radius R with impact parameter b = L/E.
type Deflection struct {
	Impact     float64
	RMin       float64
	DeltaPhi   float64
	FlatPhi    float64
	Angle      float64 // DeltaPhi − FlatPhi
	Captured   bool
	Finished   bool
	Samples    []OrbitSample
	Constraint float64
}

// LightDeflection integrates a null equatorial ray from (r = R, φ = 0) with
// E = 1 and L = b. It stops when r returns to R, or when r < rStop (capture).
func LightDeflection(m Metric, b, R, rStop float64, cfg IntegrateConfig) Deflection {
	out := Deflection{Impact: b, RMin: R}
	x := Vec{0, R, math.Pi / 2, 0}
	v, ok := VelocityFromConserved(m, x, 1, b, Null, true)
	if !ok {
		return out
	}
	cfg.Kind = Null
	cfg.Project = false
	if cfg.T1 == 0 {
		cfg.T1 = 40 * R
	}
	outPast := false
	cfg.Observe = func(s State) {
		out.Samples = append(out.Samples, OrbitSample{Lambda: s.Lambda, X: s.X, V: s.V})
		if s.X[1] < out.RMin {
			out.RMin = s.X[1]
		}
		if s.V[1] > 0 && s.X[1] < R {
			outPast = true
		}
	}
	cfg.Stop = func(s State) bool {
		if s.X[1] < rStop {
			out.Captured = true
			return true
		}
		return outPast && s.X[1] >= R && s.V[1] > 0
	}
	res := IntegrateGeodesic(m, x, v, cfg)
	out.Finished = res.Finished || res.Stopped
	if len(out.Samples) > 0 {
		last := out.Samples[len(out.Samples)-1]
		out.DeltaPhi = last.X[3] - out.Samples[0].X[3]
		if !out.Captured {
			if phi, ok := phiAtRadius(out.Samples, R, false); ok {
				out.DeltaPhi = phi - out.Samples[0].X[3]
			}
		}
		out.Constraint = Constraint(m, last.X, last.V)
	}
	if b > 0 && b < R {
		out.FlatPhi = 2 * math.Acos(b/R)
	}
	if !out.Captured {
		out.Angle = out.DeltaPhi - out.FlatPhi
	}
	return out
}

// SchwarzschildDeflectionAngle is the finite-radius deflection
// 2 ∫_rmin^R dφ − 2 arccos(b/R) for a Schwarzschild photon.
func SchwarzschildDeflectionAngle(M, b, R float64) (angle, rmin float64, ok bool) {
	rmin, ok = schwarzschildPhotonRMin(M, b)
	if !ok || rmin >= R {
		return 0, 0, false
	}
	// r = rmin + w² removes the 1/√(r−rmin) turning-point singularity.
	wMax := math.Sqrt(R - rmin)
	q0 := 2 * b * b * (rmin - 3*M) / math.Pow(rmin, 4)
	f := func(w float64) float64 {
		r := rmin + w*w
		var q float64
		if w < 1e-8*wMax {
			q = q0
		} else {
			q = (1 - (1-2*M/r)*b*b/(r*r)) / (r - rmin)
		}
		return 2 * b / (r * r) / math.Sqrt(q)
	}
	dphi := 2 * Simpson(f, 0, wMax, 4000)
	flat := 2 * math.Acos(math.Min(1, b/R))
	return dphi - flat, rmin, true
}

// phiAtRadius linearly interpolates φ to the first crossing of radius r.
// inbound selects a decreasing-r crossing.
func phiAtRadius(samples []OrbitSample, r float64, inbound bool) (float64, bool) {
	for i := 1; i < len(samples); i++ {
		r0 := samples[i-1].X[1]
		r1 := samples[i].X[1]
		cross := (inbound && r0 >= r && r1 <= r) || (!inbound && r0 <= r && r1 >= r)
		if cross && r1 != r0 {
			frac := (r - r0) / (r1 - r0)
			return samples[i-1].X[3] + frac*(samples[i].X[3]-samples[i-1].X[3]), true
		}
	}
	return 0, false
}

func schwarzschildPhotonRMin(M, b float64) (float64, bool) {
	bcrit := 3 * math.Sqrt(3) * M
	if b <= bcrit {
		return 0, false
	}
	// r² = b² (1 - 2M/r), root in (3M? actually > photon, < b).
	f := func(r float64) float64 {
		return r*r - b*b*(1-2*M/r)
	}
	return bisectSign(f, 3*M*(1+1e-6), b, 1e-10)
}

// DragResult is the azimuth gained by an equatorial geodesic between two radii.
type DragResult struct {
	DeltaPhi float64
	E, L     float64
	Finished bool
	Samples  []OrbitSample
}

// AzimuthChange integrates dφ along an inbound equatorial geodesic from r0 to r1
// and also returns the quadrature ∫ (u^φ/u^r) dr of the same constants of motion.
func AzimuthChange(m Metric, E, L, r0, r1 float64, cfg IntegrateConfig) (geo, quad DragResult) {
	x := Vec{0, r0, math.Pi / 2, 0}
	v, ok := VelocityFromConserved(m, x, E, L, Timelike, r1 < r0)
	if !ok {
		return DragResult{}, DragResult{}
	}
	geo.E, geo.L = E, L
	quad.E, quad.L = E, L
	cfg.Kind = Timelike
	inbound := r1 < r0
	cfg.Observe = func(s State) {
		geo.Samples = append(geo.Samples, OrbitSample{Lambda: s.Lambda, X: s.X, V: s.V})
	}
	cfg.Stop = func(s State) bool {
		if inbound {
			return s.X[1] <= r1
		}
		return s.X[1] >= r1
	}
	if cfg.T1 == 0 {
		cfg.T1 = 50 * math.Abs(r0-r1)
	}
	res := IntegrateGeodesic(m, x, v, cfg)
	geo.Finished = res.Stopped || res.Finished
	if len(geo.Samples) >= 2 {
		phi1, ok := phiAtRadius(geo.Samples, r1, r1 < r0)
		if !ok {
			phi1 = geo.Samples[len(geo.Samples)-1].X[3]
		}
		geo.DeltaPhi = phi1 - geo.Samples[0].X[3]
	}
	quad.DeltaPhi = azimuthQuadrature(m, E, L, r0, r1, 2000)
	quad.Finished = !math.IsNaN(quad.DeltaPhi)
	return geo, quad
}

func azimuthQuadrature(m Metric, E, L, r0, r1 float64, n int) float64 {
	if n < 2 {
		n = 2
	}
	h := (r1 - r0) / float64(n)
	// Midpoint rule. u^φ/u^r is smooth away from turning points.
	var s float64
	inbound := r1 < r0
	for i := 0; i < n; i++ {
		r := r0 + (float64(i)+0.5)*h
		x := Vec{0, r, math.Pi / 2, 0}
		v, ok := VelocityFromConserved(m, x, E, L, Timelike, inbound)
		if !ok || math.Abs(v[1]) < 1e-14 {
			return math.NaN()
		}
		s += v[3] / v[1]
	}
	return s * h
}

// InfallResult is a radial fall in Eddington–Finkelstein coordinates.
type InfallResult struct {
	Samples    []OrbitSample
	TauHorizon float64
	TauAt      []float64 // proper time at the requested radii
	RAt        []float64
	Finished   bool
	Constraint float64
}

// RadialInfall releases a particle from rest at r0 and integrates the ingoing
// Eddington–Finkelstein geodesic down to rMin.
func RadialInfall(M, r0, rMin float64, mark []float64, cfg IntegrateConfig) InfallResult {
	var out InfallResult
	x, v, ok := EFRestState(M, r0)
	if !ok {
		return out
	}
	metric := SchwarzschildEF{M: M}
	need := RadialRestTau(M, r0, math.Max(rMin, 1e-3))
	if math.IsNaN(need) || need <= 0 {
		need = 100
	}
	// Integrate at least long enough for a radial fall to reach rMin.
	if cfg.T1 < need*1.05 {
		cfg.T1 = need * 1.25
	}
	cfg.Kind = Timelike
	hitH := false
	cfg.Observe = func(s State) {
		out.Samples = append(out.Samples, OrbitSample{Lambda: s.Lambda, X: s.X, V: s.V})
		if !hitH && s.X[1] <= 2*M {
			hitH = true
			out.TauHorizon = s.Lambda
		}
	}
	cfg.Stop = func(s State) bool {
		return s.X[1] <= rMin || s.X[1] <= 0
	}
	res := IntegrateGeodesic(metric, x, v, cfg)
	out.Finished = res.Finished || res.Stopped
	if len(out.Samples) > 0 {
		last := out.Samples[len(out.Samples)-1]
		out.Constraint = Constraint(metric, last.X, last.V)
	}
	for _, rm := range mark {
		out.RAt = append(out.RAt, rm)
		out.TauAt = append(out.TauAt, interpolateTau(out.Samples, rm))
	}
	if !hitH {
		out.TauHorizon = interpolateTau(out.Samples, 2*M)
	} else {
		out.TauHorizon = interpolateTau(out.Samples, 2*M)
	}
	return out
}

func interpolateTau(s []OrbitSample, r float64) float64 {
	for i := 1; i < len(s); i++ {
		r0, r1 := s[i-1].X[1], s[i].X[1]
		if (r0-r)*(r1-r) <= 0 && r0 != r1 {
			frac := (r0 - r) / (r0 - r1)
			return s[i-1].Lambda + frac*(s[i].Lambda-s[i-1].Lambda)
		}
	}
	return math.NaN()
}

// StaticRedshift returns 1+z for a photon exchanged by static observers
// (four-velocity along ∂_t) at equatorial radii rEmit and rRecv.
// Static observers exist only where g_tt < 0.
func StaticRedshift(m Metric, rEmit, rRecv, theta float64) (float64, bool) {
	ge := m.Cov(Vec{0, rEmit, theta, 0})
	grv := m.Cov(Vec{0, rRecv, theta, 0})
	if ge[0][0] >= 0 || grv[0][0] >= 0 {
		return 0, false
	}
	return math.Sqrt((-grv[0][0]) / (-ge[0][0])), true
}
