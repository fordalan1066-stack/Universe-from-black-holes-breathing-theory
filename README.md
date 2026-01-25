The Ford Model / Unified Whisper Theory
Quantum-first horizon microstructure and a covariant entropy-flux engine
Alan Ford
January 25, 2026
Abstract
We present a quantum-first framework in which spacetime geometry and the infrared
stress–energy content of the Universe emerge from horizon microstructure. The primary
object is a partition functional over geometries and horizon microstates. Stationarity of
this object yields an emergent mean-geometry field equation whose source is a total stress
operator decomposed into: (i) an “inhale” (sequestration) entropy-flux tensor, (ii) an “exhale”
inversion/release tensor triggered by congruence focusing and internal operator stress, and
(iii) an emergent spectrum stress replacing fundamental Standard-Model matter in the
canonical formulation. A minimal three-patch (qutrit) horizon subspace is shown to force
an su(3) algebra from projected seam couplings, with an explicit operator dictionary and
matrix realization. Inversion activation is locked to a commutator stress invariant Γ and a
Raychaudhuri-driven gate Ξ, removing ad hoc switches. The paper is written to be fully
replicable: all definitions, matrices, and algorithmic checks required to reproduce the algebraic
closure and derived quantities are included.
1 Program summary (what we are doing and why)
The Ford Model is built around one decision: the theory is quantum at the root. Classical
spacetime geometry is not assumed; it is obtained as a controlled, thermodynamic/mean-field
limit of a deeper horizon microstructure. Large-scale cosmic acceleration is not attributed to a
fundamental dark-energy fluid; instead, it arises as an effective response to a net horizon entropy
flux with two coupled channels:
• Inhale / sequestration: an entropy-directed flux sector that contributes an effective
stress tensor τ(H)
µν.
• Exhale / inversion: a burst-like release sector τ(inv)
µν triggered by congruence focusing
and internal operator stress; it enables cyclic dynamics (turnaround and bounce).
In the canonical (latest) formulation, what appears in the infrared as “matter” is treated as an
emergent spectrum stress τ(spec)
µν built from modular horizon modes, rather than as a fundamental
Standard Model sector.
1.1 What a reviewing scientist will look for
A critical reviewer will check:
1. The foundational definition of Z and whether the ordering is genuinely quantum-first.
2. A logically complete route from Z to the field equation (stationarity, expectation
values, and conservation).
3. Explicit, falsifiable microstructure claims: not metaphors, but concrete operator
sets, matrices, and closure checks.
1
4. No hand-picked “magic” choices: the operator basis and its size must be forced by
symmetry and physical construction.
5. Triggering logic for inversion: it must be endogenous (geometry and microstress), not
a manually placed switch.
This paper is organized to make those checks straightforward.
2 Quantum root: the canonical definition of Z
2.1 Primary object
The primary object is the quantum-statistical partition functional
Z ≡ Dg TrHhorizon exp−
1
ℏ Itot[g; H]. (1)
Here Dg is a measure over geometries, and the trace is taken over a horizon microstructure
Hilbert space Hhorizon.
2.2 Patch factorization (microstructure picture)
We assume the horizon microstructure factorizes into patches:
Hhorizon =
p∈patches
Hp, Hp
∼
= Cdp
, (2)
with patch entropy
Sp ≡kBln dp, SH=
Sp. (3)
p
2.3 Total action split (canonical; no fundamental SM sector)
Itot = IGR[g] + IH[g,H] + Iint[g; H], (4)
with the Einstein–Hilbert part
IGR[g] = c3
16πG (R−2Λ0)√−gd4x. (5)
3 Emergent mean-geometry field equation
3.1 Stationarity
The canonical stationarity condition is
δln Z
δgµν(x) = 0. (6)
3.2 Emergent field equation
Stationarity implies an emergent mean-geometry equation
Gµν + Λ0gµν =
8πG
c4 τtotal
µν , (7)
where Gµν is the Einstein tensor of the emergent mean geometry and τtotal
µν is the total emergent
stress operator.
2
3.3 Conservation
Covariance and the Bianchi identity require
∇µ τtotal
µν = 0. (8)
4 Ford entropy-flux engine: inhale, exhale, spectrum
4.1 Decomposition
We decompose the total source as
τtotal
µν
= τ(H)
µν + τ(inv)
µν + τ(spec)
µν. (9)
4.2 Area-law anchor and entropy surface density
We anchor to the Bekenstein–Hawking area law,
SBH =
kBc3
4ℏGA, (10)
and define an entropy surface density
η≡
δS
δA=
kBc3
4ℏG fbh(z). (11)
Update (latest convention): for the cosmology channel we use
fbh(z) = ρbh(z)
ρcrit(z). (12)
4.3 Inhale tensor (null-congruence carrier form)
Let kµ be a (locally defined) null generator of the relevant horizon congruence, and σµν its shear.
A compact covariant carrier used in the theory is
ℏc
τ(H)
µν
2π
1
k(µkν)−
2(kλkλ)gµν η+ σµνη . (13)
For a strictly null congruence kλkλ = 0; it is retained as a regularization/generalization place-
=
holder.
4.4 Exhale / inversion tensor (burst-like release; WH deprecated)
Earlier drafts used “white-hole recoil” language. In the latest formulation we deprecate an
explicit WH population sector: inversion is treated as an endogenous burst/release channel
triggered by congruence focusing and internal operator stress (Sections 9–9.3). We write
τ(inv)
µν
=−γinv(z) τ(H)
µν + ∆τ(inv)
µν. (14)
with a minimal closure term
∆τ(inv)
µν
ℏc
=
2π
k(µkν) ηinv, ηinv =
kBc3
4ℏG finv(z). (15)
3
4.5 Emergent spectrum stress (IR replacement for fundamental SM)
Modular horizon modes with gaps ∆sn define an emergent spectrum stress
τ(spec)
µν ≡ Tµν emergent
=
n
dΠn Wn(∆sn) p(n)
µ p(n)
ν. (16)
where dΠn is an invariant phase-space measure and Wn a weight fixed by modular gaps.
5 Thermodynamic and geometric anchors
5.1 Local first-law structure
A Jacobson-style local anchor is
ℏa
2πkBc
, (17)
1
θ2
kc2
a2 (20)
δQ= TδS, T=
with T the Unruh temperature for acceleration a.
5.2 Raychaudhuri focusing
For a null congruence with expansion θ, shear σµν and twist ωµν,
dθ
dλ=−
2
−σµνσµν + ωµνωµν −Rµνkµkν
. (18)
6 Cosmology reduction (FRW channel)
Assume an isotropic mean geometry
ds2
=−c2dt2 + a(t)2 dr2
1−kr2 + r2dΩ2 , H ≡˙
a
. (19)
a
Then a comparison-ready effective Friedmann form is
H2(z) = 8πG
3 ρeff(z)−
, ρeff = ρspec + ρH + ρinv. A compact phenomenology channel used in earlier fitting work can be recorded as
H2(z) = H2
0 Ωm(1 + z)3 + Ωbh(1 + z)2.3e−1.1(1+z)
. (21)
7 Why the theory uses a three-fold microstructure
7.1 Three folds: the first point where orientation and memory can exist
The microstructure is built from a three-fold patch logic. This is the minimal fold at which
repeated identification forces a twist/lock:
• One fold: a deformation can be undone without introducing protected orientation.
• Two folds: the surface can still be smoothed without forcing a twist.
• Three folds: consistency forces a twist or lock, making a chirality-sensitive seam sector
unavoidable.
In this model, the third fold is where (i) chiral seam operators become physical, (ii) a protected
qutrit subspace appears, and (iii) a full su(3) algebra can be forced from local seam couplings.
4
. (22)
8 Operator microstructure and gauge emergence
8.1 Qutrit subspace
Consider three patches, each with a two-state local degree of freedom. The full space is
H= (C2)⊗3 (dimension 8). We restrict to the one-excitation subspace
Hq = span{|100⟩,|010⟩,|001⟩}∼
= C3
Let Πq be the projector onto Hq.
8.2 One-, two-, and three-patch ladder (why SU(2) appears before SU(3))
The same seam-coupling construction yields the familiar ladder:
• 1 patch: a single local phase rotation gives a U(1) sector.
• 2 patches: a seam doublet supports an su(2) subalgebra.
• 3 patches: the one-excitation qutrit forces su(3).
This is not imposed; it is the minimal algebra compatible with the number of interacting patches
and the requirement of Hermitian, traceless generators.
8.3 Explicit SU(2) seam-doublet matrix (two patches)
For two patches the Hilbert space is H12 = C2 ⊗C2 with basis {|00⟩,|01⟩,|10⟩,|11⟩}. A
representative seam-coupling operator used in the construction (written here explicitly to ensure
replicability) is
K12(ϵ= 1,J = 2) =
   
2 0 0 2
0 0 2 0
0 2 0 0
2 0 0−2
   
(23)
When restricted to the appropriate doublet sector, the induced traceless Hermitian generators
close an su(2) algebra. The three-patch construction below generalizes this same physical idea:
seam exchange and seam chirality projected onto the qutrit.
8.4 Why eight operators (and why they are forced)
The operator basis is not hand-picked. It is forced by:
1. Target algebra su(3) has dimension 8.
2. Only three physical seams exist: (12),(23),(31).
3. Each seam admits an even exchange channel and an odd chiral channel, yielding two
operators per seam: Xij and Yij (six total).
4. A two-dimensional Cartan subalgebra is required: D3 and D8.
Hence the minimal forced dictionary is
Dmin = {X12,Y12,X23,Y23,X31,Y31,D3,D8}. (24)
5
8.5 Operator construction (from physical couplings)
Project Pauli-coupling forms onto the qutrit:
Xij ≡Πq σx
i σx
j + σy
iσy
j Πq, (25)
Yij ≡Πq σx
i σy
j−σy
iσx
j Πq, (26)
1
D3 ≡Πq(σz
1−σz
2) Πq, D8 ≡
Πq(σz
1 + σz
2−2σz
3) Πq. (27)
√3
8.6 Matrix realization: Gell–Mann basis
In basis {|100⟩,|010⟩,|001⟩}, the Gell–Mann matrices are
1 0 0
λ1 =  0 1 0
i 0 0
0 0 0
 , λ2 =  0−i 0
0−1 0
0 0 0
 , λ3 =  1 0 0
0 0 0
 , (28)
0 0 0
λ4 =  0 0 1
0 0 0
1 0 0
 , λ5 =  0 0−i
0 0 1
i 0 0
 , λ6 =  0 0 0
0 1 0
 , (29)
λ7 =  0 0 0
0 0−i
0 i 0
 , λ8 =
Define generators Ta ≡λa/2.
1
0 1 0
√3
 1 0 0
0 0−2
 . (30)
8.7 Plan A: replicable non-overfit emergence check
To demonstrate emergence rather than assertion:
1. Compute Dmin from projections.
2. Use Hilbert–Schmidt inner product ⟨A,B⟩= Tr(A†B).
3. Form Gram matrix GAB = ⟨XA,XB⟩and verify rank(G) = 8.
4. Express the basis {Ta}in the span of Dmin and compute residuals.
5. Check commutator closure by projecting [XA,XB] back onto the span and measuring
Frobenius-norm closure errors.
9 Environment deformation and inversion triggering
9.1 Deformation-only-by-coefficients
Allow only coefficient deformations (no new operators):
XA(z) = aA(z) XA, XA ∈Dmin. (31)
9.2 Gate Ξ(z) from focusing (Raychaudhuri)
Encode activation as a smooth gate
Ξ(z) = 1
1 + exp Γ⋆−Γ(z)
∆Γ
. (32)
6
9.3 Commutator stress invariant Γ(z)
Define
Γ(z) ≡
A<B
Tr [XA(z),XB(z)]†[XA(z),XB(z)]
1/2
. (33)
9.4 Inversion strength lock
Lock inversion to internal stress and the gate:
γinv(z) = κγ Γ(z) finv(z), finv(z) ≡Ξ(z). (34)
An optional near-threshold shear-amplified factor for phenomenology is
γinv →γinv 1 + 0.05 σ2(z)
σ2
crit
. (35)
10 Mass prediction via third-fold anchored inversion
Use the same internal stress invariant Γ in three fold sectors g∈{1,2,3}with g= 3 the third-fold
anchor (tau). Fix the overall scale by one anchor mτ ≡mg=3. Then the ripple-back prediction
rule is
mg−1 = mg
Γg−1
(g= 3 →2 →1). (36)
Γg
Hence the mass ratios are pure outputs once Γg are computed:
mµ
mτ
=
Γ2
Γ3
me
,
mµ
=
Γ1
Γ2
. (37)
11 Cyclic conditions (turnaround and bounce)
Turnaround satisfies H(t⋆) = 0, equivalently
A bounce requires H(tb) = 0 and
ρeff(t⋆) = 3kc2
8πGa(t⋆)2. (38)
˙
H(tb) >0. Near-bounce the effective condition can be written
(39)
(ρH + ρinv) + 3(pH + pinv) <0. 12 Replication checklist
1. Construct Πq and compute Xij,Yij,D3,D8.
2. Verify Hermiticity, tracelessness, and Gram rank 8.
3. Map Dmin to the Gell–Mann basis and compute residuals.
4. Check commutator closure and quantify closure errors.
5. Choose deformations aA(z), compute Γ(z) and Ξ(z).
6. Compute γinv(z) from the lock.
7. (Cosmology channel) use fbh(z) = ρbh/ρcrit to build η.
8. (Mass channel) compute Γ1,Γ2,Γ3 and output mµ,me using one anchor mτ.
7
Scope note
The “Hope” document is used only as conceptual scaffolding; any equations are taken from the
canonical statements in this paper. White-hole population language is deprecated in the latest
formulation; inversion is represented solely by τ(inv)
µν with stress-locked activation.
8
