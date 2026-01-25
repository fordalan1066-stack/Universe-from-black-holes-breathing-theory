\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{bm}
\usepackage{array}
\usepackage{hyperref}
\usepackage{physics}
\usepackage{microtype}

\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}

\title{The Ford Model / Unified Whisper Theory\\
\large Quantum-first horizon microstructure and a covariant entropy-flux engine}
\author{Alan Ford}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
We present a quantum-first, horizon-microstructure framework in which mean spacetime geometry and effective matter arise as thermodynamic limits of horizon degrees of freedom. The primary object is a partition functional over geometries and horizon microstates. A covariant entropy-flux engine (the Ford Entropy Flux Tensor) provides an ``inhale/exhale'' mechanism capable of driving late-time expansion without dark energy and enabling cyclic dynamics via an inversion channel. Gauge structure and particle-sector observables are treated as emergent spectral excitations of modular horizon operators rather than fundamental Standard Model fields. We provide a minimal, non-overfit SU(3) operator dictionary on a qutrit subspace and define commutator-stress invariants used to lock inversion strength to geometric focusing and modular thresholds.
\end{abstract}

\section{Quantum Root and Canonical Ordering}

\subsection{Primary partition functional}
The Ford Model is quantum at the root. The primary object is the partition functional over geometries and horizon microstates:
\begin{equation}
Z \;\equiv\; \int \mathcal D g \;\;\mathrm{Tr}_{\mathcal H_{\text{horizon}}}\;
\exp\!\left[-\frac{1}{\hbar}\, I_{\text{tot}}[g;\,\mathcal H]\right].
\label{eq:Z}
\end{equation}

\subsection{Horizon Hilbert-space factorization (patch picture)}
We assume the horizon microstructure factorizes into patches:
\begin{equation}
\mathcal H_{\text{horizon}} \;=\; \bigotimes_{p\in \text{patches}} \mathcal H_{p},
\qquad
\mathcal H_{p}\cong \mathbb C^{d_p}.
\end{equation}
Define patch entropies
\begin{equation}
S_p \;\equiv\; k_B \ln d_p
\qquad\Rightarrow\qquad
S_{\mathcal H}=\sum_p S_p.
\end{equation}

\subsection{Total action split (no SM sector)}
The total action is decomposed as
\begin{equation}
I_{\text{tot}} \;=\; I_{\text{GR}}[g] \;+\; I_{\mathcal H}[g,\mathcal H] \;+\; I_{\text{int}}[g;\mathcal H],
\end{equation}
with Einstein--Hilbert term
\begin{equation}
I_{\text{GR}}[g] \;=\; \frac{c^3}{16\pi G}\int (R -2\Lambda_0)\sqrt{-g}\,d^4x.
\end{equation}

\section{Emergent Field Equation from Stationarity}

\subsection{Stationarity condition}
Mean geometry emerges from stationarity of $\ln Z$:
\begin{equation}
\frac{\delta \ln Z}{\delta g^{\mu\nu}(x)} \;=\; 0.
\label{eq:stationarity}
\end{equation}

\subsection{Emergent mean-geometry equation}
The canonical emergent equation is
\begin{equation}
\left\langle \widetilde G_{\mu\nu} + \Lambda_0 g_{\mu\nu} \right\rangle
\;=\;
\frac{8\pi G}{c^4}\left\langle \widehat{\tau}^{\text{total}}_{\mu\nu}\right\rangle.
\label{eq:emergentEinstein}
\end{equation}
Covariance implies conservation:
\begin{equation}
\nabla^\mu \left\langle \widehat{\tau}^{\text{total}}_{\mu\nu}\right\rangle \;=\; 0.
\label{eq:conservation}
\end{equation}

\section{Ford Entropy Flux Tensor (Breathing Engine)}

We decompose the total emergent stress-energy operator as
\begin{equation}
\widehat{\tau}^{\text{total}}_{\mu\nu}
\;=\;
\widehat{\tau}^{(H)}_{\mu\nu}
\;+\;
\widehat{\tau}^{(\text{inv})}_{\mu\nu}
\;+\;
\widehat{\tau}^{(\text{spec})}_{\mu\nu}.
\label{eq:taudecomp}
\end{equation}

\subsection{Entropy density anchor}
Bekenstein--Hawking area law:
\begin{equation}
S_{BH} \;=\; \frac{k_B c^3}{4\hbar G}\,A.
\end{equation}
Define entropy surface density
\begin{equation}
\eta \;\equiv\;\frac{\delta S}{\delta A}
\;=\; \frac{k_B c^3}{4\hbar G}\; f_{bh}(z).
\label{eq:eta}
\end{equation}

\subsection{Inhale tensor (null-congruence form)}
Let $k^\mu$ be a null generator and $\sigma_{\mu\nu}$ the shear. Define
\begin{equation}
\boxed{
\widehat{\tau}^{(H)}_{\mu\nu}
=
\frac{\hbar c}{2\pi}
\left[
\left(k_{(\mu}k_{\nu)} - \frac12 (k^\lambda k_\lambda) g_{\mu\nu}\right)\eta
\;+\;
\sigma_{\mu\nu}\eta
\right].
}
\label{eq:tauH}
\end{equation}

\subsection{Inversion / exhale sector}
We define the inversion channel as
\begin{equation}
\boxed{
\widehat{\tau}^{(\text{inv})}_{\mu\nu}
=
-\gamma_{\text{inv}}(z)\,\widehat{\tau}^{(H)}_{\mu\nu}
+
\Delta \widehat{\tau}^{(\text{inv})}_{\mu\nu}.
}
\label{eq:tauinv}
\end{equation}
A minimal closure is
\begin{equation}
\Delta \widehat{\tau}^{(\text{inv})}_{\mu\nu}
=
\frac{\hbar c}{2\pi}\left(k_{(\mu}k_{\nu)}\right)\eta_{\text{inv}},
\qquad
\eta_{\text{inv}}=\frac{k_B c^3}{4\hbar G}\, f_{\text{inv}}(z).
\label{eq:tauinvclosure}
\end{equation}

\subsection{Emergent spectrum stress tensor}
Emergent matter is encoded as spectral stress
\begin{equation}
\widehat{\tau}^{(\text{spec})}_{\mu\nu}
\equiv
\left\langle \widehat{T}_{\mu\nu}\right\rangle_{\text{emergent}}
=
\sum_{n}\int d\Pi_n\;\; \mathcal W_n \;\;
p^{(n)}_\mu p^{(n)}_\nu,
\qquad
\mathcal W_n=\mathcal W(\Delta s_n).
\label{eq:tauspec}
\end{equation}

\section{Thermodynamic and Geometric Anchors}

\subsection{Horizon first law (Jacobson-style)}
\begin{equation}
\delta Q \;=\; T\,\delta S,
\qquad
T \;=\; \frac{\hbar a}{2\pi k_B c}.
\end{equation}

\subsection{Raychaudhuri focusing}
For a null congruence,
\begin{equation}
\frac{d\theta}{d\lambda} =
-\frac12 \theta^2
-\sigma_{\mu\nu}\sigma^{\mu\nu}
+\omega_{\mu\nu}\omega^{\mu\nu}
-R_{\mu\nu}k^\mu k^\nu.
\label{eq:raychaudhuri}
\end{equation}

\section{Cosmological Reduction (FRW)}

Assume FRW mean geometry
\begin{equation}
ds^2=-c^2dt^2+a(t)^2\left(\frac{dr^2}{1-kr^2}+r^2d\Omega^2\right),
\qquad
H\equiv \frac{\dot a}{a}.
\end{equation}
Effective Friedmann form:
\begin{equation}
\boxed{
H^2(z)=\frac{8\pi G}{3}\rho_{\text{eff}}(z)-\frac{kc^2}{a^2},
\qquad
\rho_{\text{eff}}=\rho_{\text{spec}}+\rho_H+\rho_{\text{inv}}.
}
\end{equation}

\subsection{Working phenomenology channel}
\begin{equation}
\boxed{
H^2(z)=H_0^2\left[\Omega_m(1+z)^3+\Omega_{bh}(1+z)^{2.3}e^{-1.1(1+z)}\right].
}
\label{eq:Hfit}
\end{equation}

\section{Core Functions}

\subsection{$f_{bh}(z)$ from population-driven horizon area}
Define a BH-area proxy
\begin{equation}
\mathcal A_{bh}(z)\propto \int dM\,n(M,z)\,M^2,
\end{equation}
and normalize
\begin{equation}
f_{bh}(z)\equiv \frac{\mathcal A_{bh}(z)}{\mathcal A_{bh}(0)}.
\end{equation}

\subsection{Inversion activation and breathing diagnostic}
\begin{equation}
f_{\text{inv}}(z)=\mathcal F\!\left(f_{bh}(z),\Xi(z)\right),
\qquad
\mathcal B(z)=f_{bh}(z)-\lambda f_{\text{inv}}(z).
\end{equation}

\section{Cyclic Conditions}

\subsection{Turnaround}
\begin{equation}
H(t_\star)=0
\quad\Longleftrightarrow\quad
\rho_{\text{eff}}(t_\star)=\frac{3kc^2}{8\pi G\,a(t_\star)^2}.
\end{equation}

\subsection{Bounce}
\begin{equation}
H(t_b)=0,\qquad \dot H(t_b)>0,
\end{equation}
with an effective condition near bounce
\begin{equation}
\boxed{
\left(\rho_H+\rho_{\text{inv}}\right)+3\left(p_H+p_{\text{inv}}\right)<0.
}
\end{equation}

\section{Minimal SU(3) Emergence on the Qutrit}

\subsection{Qutrit subspace}
Let the three-patch Hilbert space be $(\mathbb C^2)^{\otimes 3}$ and restrict to the one-excitation sector
\begin{equation}
\mathcal H_q=\mathrm{span}\{|100\rangle,|010\rangle,|001\rangle\}\cong\mathbb C^3.
\end{equation}
Let $\Pi_q$ be the projector onto $\mathcal H_q$.

\subsection{Minimal operator dictionary}
Define the minimal forced dictionary
\begin{equation}
\mathcal D_{\min}=\{X_{12},Y_{12},X_{23},Y_{23},X_{31},Y_{31},D_3,D_8\}.
\end{equation}
The operators arise from projected Pauli couplings:
\begin{align}
X_{ij} &\equiv \Pi_q\left(\sigma_i^x\sigma_j^x+\sigma_i^y\sigma_j^y\right)\Pi_q,\\
Y_{ij} &\equiv \Pi_q\left(\sigma_i^x\sigma_j^y-\sigma_i^y\sigma_j^x\right)\Pi_q,\\
D_3 &\equiv \Pi_q\left(\sigma_1^z-\sigma_2^z\right)\Pi_q,\\
D_8 &\equiv \frac{1}{\sqrt 3}\Pi_q\left(\sigma_1^z+\sigma_2^z-2\sigma_3^z\right)\Pi_q.
\end{align}

\subsection{Gell--Mann matrices}
We identify the standard SU(3) generators $\lambda_a$ on $\mathcal H_q$:
\begin{align}
\lambda_1&=\begin{pmatrix}0&1&0\\1&0&0\\0&0&0\end{pmatrix},&
\lambda_2&=\begin{pmatrix}0&-i&0\\ i&0&0\\0&0&0\end{pmatrix},&
\lambda_3&=\begin{pmatrix}1&0&0\\0&-1&0\\0&0&0\end{pmatrix},\\[6pt]
\lambda_4&=\begin{pmatrix}0&0&1\\0&0&0\\1&0&0\end{pmatrix},&
\lambda_5&=\begin{pmatrix}0&0&-i\\0&0&0\\ i&0&0\end{pmatrix},&
\lambda_6&=\begin{pmatrix}0&0&0\\0&0&1\\0&1&0\end{pmatrix},\\[6pt]
\lambda_7&=\begin{pmatrix}0&0&0\\0&0&-i\\0&i&0\end{pmatrix},&
\lambda_8&=\frac{1}{\sqrt 3}\begin{pmatrix}1&0&0\\0&1&0\\0&0&-2\end{pmatrix}.
\end{align}

\section{Operator Stress and Inversion Lock}

\subsection{Commutator stress invariant}
Define the commutator stress invariant for a deformed operator set $\{\widetilde T^a\}$:
\begin{equation}
\Gamma
\;\equiv\;
\left(
\sum_{a<b}
\mathrm{Tr}
\Big(
[\widetilde T^{\,a},\widetilde T^{\,b}]^\dagger
[\widetilde T^{\,a},\widetilde T^{\,b}]
\Big)
\right)^{1/2}.
\label{eq:Gamma}
\end{equation}

\subsection{Inversion strength lock}
We lock inversion strength to modular stress and activation:
\begin{equation}
\boxed{
\gamma_{\rm inv}(z)=\kappa_\gamma\,\Gamma(z)\,f_{\rm inv}(z).
}
\end{equation}

\subsection{Environment deformation (coefficients only)}
Let the environment modulate coefficients on a fixed dictionary:
\begin{equation}
\widetilde X_A(z)=a_A(z)\,X_A,\qquad X_A\in\mathcal D_{\min}.
\end{equation}
A smooth activation gate may be defined by a logistic trigger in $\Gamma(z)$:
\begin{equation}
\Xi(z)=\frac{1}{1+\exp\left(\frac{\Gamma_\star-\Gamma(z)}{\Delta\Gamma}\right)}.
\end{equation}

\section{Lepton Masses from Horizon Operator Stress}

\subsection{Mass scaling from commutator stress}
With one anchor mass $m_\tau$,
\begin{equation}
m_x = m_\tau \sqrt{\frac{\Gamma_x}{\Gamma_3}}.
\end{equation}

\section{Timing Calibration Layer (Non-fundamental)}
We record the calibration values used for breath-phase coherence:
\begin{equation}
\nu_{\text{tone}}=42.00003862,\qquad
\nu_{\text{osc}}\approx 10000003862,\qquad
c_{\text{eff}}=c\cdot 1.000003862.
\end{equation}

\section{References}
(References to be inserted.)

\end{document}
