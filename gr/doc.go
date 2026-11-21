// Copyright 2026 The Real Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gr is a general-relativity simulator in geometric units (G = c = 1).
//
// The metric signature is (−, +, +, +). Black-hole coordinates are
// (t, r, θ, φ) unless a metric says otherwise. Curvature is computed from the
// metric and its first derivatives: Christoffel symbols are exact when the
// metric supplies derivatives, and the Riemann tensor differentiates those
// symbols numerically. Timelike and null geodesics are integrated with an
// adaptive Dormand–Prince method.
package gr
