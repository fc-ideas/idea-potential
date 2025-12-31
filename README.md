
# Designing Paradigms for Long-Term Progress
### Research as Reachability: A Functorial Account of Scientific Progress

In plain language, this subtitle says the paper treats progress as what ideas can reach when you apply allowed transformations.

## Abstract

In plain language, this abstract says some ideas keep producing results because there are many valid ways to transform them; the paper counts that reach and keeps human goal-setting, building, and interpretation in view.

Here we offer an account of why some research ideas keep producing new work while others stall. We treat each idea as something that can be transformed in many ways, and ask how many strong, valid follow-on ideas those transformations can reach. The framework also treats human input as central: people set the goals, build the systems, and interpret the results. The result is a structured way to compare paradigms and to explain why some are easier to extend, reuse, and improve over time.

## 1. Introduction: Beyond Performance and Expressivity

In plain language, this introduction says fertility means many extensions, reusable tools, diagnostics, and steering interfaces, and that standard metrics miss it. It uses neural networks versus kernels to show the gap, then introduces the framework, its human component, and its goal of selecting and diagnosing paradigms.

Scientific progress is uneven. Some research ideas rapidly generate entire ecosystems of methods, diagnostics, and applications, while others -- despite elegance or strong theoretical guarantees -- lead to narrow or short-lived lines of work. Scientific history repeatedly shows that paradigms succeed not merely because they are correct or optimal, but because they are *fertile*. Fertility manifests as:

* many viable extensions,
* reusable conceptual tools,
* diagnostics that guide exploration,
* and interfaces that allow humans to steer development.

This difference is rarely captured by standard evaluation criteria such as performance benchmarks, expressivity, or asymptotic optimality. As a result, promising paradigms are often indistinguishable from dead ends until years later. Machine learning provides a striking example. Kernel methods and neural networks are comparable in expressive power under broad conditions, yet only neural networks have sustained decades of compounding progress under current hardware and data regimes. The usual explanations -- data availability, compute, or optimization tricks -- describe *what happened*, but not *why certain paradigms remain fertile while others stagnate*. What is missing is a structural account of generativity that does not assume a single learning style.

This paper proposes such an account. We introduce a formal framework in which research ideas are treated as artifacts indexed by semantic commitments, and methodological moves -- including discrete rewrites, inference steps, and theory revision -- are treated as composable transformations. Within this framework, the central quantity is **generative potential**: the measure of high-quality, valid ideas reachable from a given idea through admissible compositions of transformations, given a fixed validity predicate and quality functional.

A key aspect of the framework is the explicit inclusion of the human researcher. Generative progress depends not only on what transformations are possible, but on which transformations are legible, predictable, and reconstructible by humans. We therefore treat specification, realization, and interpretation as first-class components, formalizing the bidirectional coupling between ideas and human cognition when these channels can be modeled.

The goal of this work is not retrospective explanation alone. Our primary aim is prospective: to provide a **structural, formal method** for:

1. selecting paradigm candidates with high long-term potential, and
2. diagnosing what is missing when a paradigm fails to generate progress.

We show that many familiar machine-learning techniques arise from a small set of primitive generators, and that successful paradigms are those that expose these generators cleanly, support their composition, and admit stable abstraction and revision mechanisms. Using neural networks and kernel machines as a case study, and extending the analysis to neurosymbolic systems, we argue that progress is driven less by isolated breakthroughs than by the structure of the idea space itself. Paradigms succeed when they make that space navigable by humans as much as by algorithms, subject to the validity constraints that define the domain.

---

## 2. Assumptions and Semantic Base

In plain language, this section states the modeling assumptions and defines how meanings and concrete artifacts are organized so the rest of the framework is well-posed.

### 2.1 Critical assumptions

In plain language, this subsection says we must be able to formalize meanings, map them to matching artifacts, define allowed moves, agree on validity and quality, compare under the same settings, and model human specification, realization, and interpretation when we claim human-guided reachability.

The framework is conditioned on the following assumptions:

* **Semantic representation**: semantic commitments can be modeled as objects and morphisms in a category $\mathcal{S}$. Plainly: we can describe the meaning, constraints, and goals of a task in a formal way and describe how those meanings change.
* **Fibered artifacts**: implementable artifacts can be modeled in fibers $\mathcal{C}_s$ indexed by $\mathcal{S}$. Plainly: for each set of assumptions, there is a matching set of concrete models and methods that belong with it.
* **Typed transformations**: admissible transformations (including rewrites) are typed and composable in a way that supports reachability. Plainly: only certain moves are allowed between ideas, and we can chain them without ambiguity.
* **Validity and quality**: a domain-specific validity predicate $\mathrm{Valid}$ and quality functional $\mathcal{Q}$ are defined and stable across the comparisons being made. Plainly: we agree on what counts as correct and what counts as good, and those rules do not change mid-comparison.
* **Comparability**: claims about relative generativity only compare paradigms under the same choices of $\mathbb{F}$, $\mathrm{Valid}$, $\mathcal{Q}$, and $\mu$. Plainly: we do not compare systems using different rules, scoring functions, or measures.
* **Human channels**: specification, realization, and interpretation can be operationalized for the domain; otherwise human-conditioned reachability is out of scope. Plainly: we can model how people set goals, build systems, and interpret outputs; if we cannot, we do not claim human-steered reachability.

### 2.2 Semantic base and fibered idea space

In plain language, this subsection says each set of semantic commitments has its own matching space of concrete artifacts, and changing the commitments moves you between those spaces. An idea is a concrete package of model, data, objectives, training, and evaluation, and reachability is computed over these typed moves.

Let $\mathcal{S}$ be a category of semantic commitments (tasks, invariances, constraints, causal assumptions, correctness criteria). For each $s \in \mathrm{Ob}(\mathcal{S})$, define a fiber $\mathcal{C}_s$ of implementable artifacts consistent with $s$. A change of semantics $u:s\to s'$ induces a reindexing functor:

$$u^*:\mathcal{C}_{s'} \to \mathcal{C}_s$$

An idea-state lives in a fiber. For $X \in \mathcal{C}_s$, we may write:

$$X = (\text{model}, \text{representation}, \text{objective}, \text{data}, \text{training}, \text{evaluation})$$

Research progress corresponds to typed transformations within and across fibers. Reachability is computed on the resulting typed graph of transformations, not just on a single endofunctor monoid.

---

## 3. Generative Transformations and Reachability

In plain language, this section defines the allowed kinds of moves (within a semantic setting, across settings, and via discrete rewrites) and then defines reachability and generative potential on top of them.

Let $\mathbb{F}$ be the set of admissible transformations, with typed domains and codomains. We distinguish:

* **Within-fiber endofunctors** $F_s:\mathcal{C}_s \to \mathcal{C}_s$ (implementation-level moves).
* **Cross-fiber functors** $T_{s \to s'} : \mathcal{C}_s \to \mathcal{C}_{s'}$ (semantic translations and structural leaps).
* **Rewrite/search operators** that act as 2-morphisms or rewrite rules between morphisms, capturing discrete proof steps, program transforms, and evolutionary operators.

Let $\langle \mathbb{F} \rangle$ denote well-typed compositions generated by these transformations.

### 3.1 Reachability

In plain language, Reach_F(X) is the set of all ideas Y, across all semantic fibers, that you can obtain by chaining allowed transformations from X, treating isomorphic results as the same.

For an idea $X\in\mathcal{C}_s$, define the reachable set across all fibers:

$$\mathrm{Reach}_{\mathbb{F}}(X) := \{\, Y \in \mathrm{Ob}(\bigsqcup_{s}\mathcal{C}_s) \mid \exists F \in \langle \mathbb{F} \rangle \text{ such that } F(X) \cong Y \,\}$$

These are the ideas systematically derivable from $X$ by admissible research moves, including discrete rewrites and cross-fiber translations.

### 3.2 Quality and validity-aware reachability

In plain language, we only count reachable ideas that clear a quality bar Q(Y) >= tau and still satisfy the validity constraints Valid(Y).

Let $\mathcal{Q}:\bigsqcup_s \mathcal{C}_s \to \mathbb{R}\cup\{-\infty\}$ be a quality functional and let $\tau$ be a threshold. Let $\mathrm{Valid}:\bigsqcup_s \mathcal{C}_s\to\mathbf{Bool}$ capture semantic validity (soundness, calibration, constraint satisfaction). Define:

$$\mathrm{Reach}_{\mathbb{F}}^{\tau}(X) := \{\, Y \in \mathrm{Reach}_{\mathbb{F}}(X) \mid \mathcal{Q}(Y) \ge \tau \ \wedge\ \mathrm{Valid}(Y) \,\}$$

This ensures generativity is measured on a valid, semantically constrained subspace.

### 3.3 Generative potential

In plain language, generative potential is the size of that filtered set measured by a chosen measure mu (count, growth, volume, or entropy), and comparisons only make sense when F, Valid, Q, and mu are fixed.

The generative potential of $X$ relative to $\mathbb{F}$ and $\tau$ is:

$$\mathrm{GP}_{\mathbb{F}}^{\tau}(X) := \mu\!\left(\mathrm{Reach}_{\mathbb{F}}^{\tau}(X)\right)$$

where $\mu$ is a measure on subsets of $\bigsqcup_s \mathrm{Ob}(\mathcal{C}_s)$ (cardinality, growth rate by depth, volume under an embedding, or entropy over equivalence classes).

Comparisons across paradigms are meaningful only when $\mathbb{F}$, $\mathrm{Valid}$, $\mathcal{Q}$, and $\mu$ are held fixed.

### 3.4 Depth-sensitive refinement

In plain language, this version looks only at chains up to length k and weights them by w_k, so short, reliable derivations count more than long, fragile ones.

Let $\langle \mathbb{F} \rangle_{\le k}$ denote compositions of length at most $k$, and define $\mathrm{Reach}_{\mathbb{F},k}^{\tau}(X)$ analogously. A depth-sensitive version is:

$$\mathrm{GP}_{\mathbb{F}}^{\tau}(X) = \sum_{k=0}^{\infty} w_k \, \mu\!\left(\mathrm{Reach}_{\mathbb{F},k}^{\tau}(X)\right)$$

with $w_k$ a decay weight.

> Generative potential is the measure of the set of high-quality, valid ideas reachable from an idea via admissible compositions of generative transformations, including cross-fiber translations.

---

## 4. Human Coupling as Specification, Realization, Interpretation

In plain language, people specify semantics (Spec), realize artifacts (Realize_s), and interpret outputs (Explain_s); transformations that factor through these channels define F_h, and GP under F_h captures human-conditioned reachability.

We treat human interaction as three distinct channels rather than a single "nudging" interface:

* **Specification** $\mathrm{Spec}:\mathcal{H}\to\mathcal{S}$ defines semantic commitments (constraints, priors, ontologies).
* **Realization** $\mathrm{Realize}_s:\mathcal{H}\to\mathcal{C}_s$ builds implementable artifacts consistent with those commitments.
* **Interpretation** $\mathrm{Explain}_s:\mathcal{C}_s\to\mathcal{H}$ returns human-usable structure (rules, proofs, summaries, counterexamples).

Define the human-accessible transformations as those that factor through human interpretation and realization:

$$\mathbb{F}_h := \{\, F:\mathcal{C}_s\to\mathcal{C}_{s'} \mid \exists G:\mathcal{H}\to\mathcal{H} \text{ with } F \cong \mathrm{Realize}_{s'} \circ G \circ \mathrm{Explain}_s \,\}$$

Then the effective, human-conditioned generative potential is:

$$\mathrm{GP}_{\mathbb{F}}^{\tau}(X \mid \mathcal{H}) := \mu\!\left(\mathrm{Reach}_{\mathbb{F}_h}^{\tau}(X)\right)$$

A paradigm is practically generative only if it supports the human channels required by the domain with stable, composable interfaces; otherwise human-conditioned reachability is limited.

---

## 5. Algebraic Structure of Research Transformations

In plain language, this section explains how transformations behave when repeated, reordered, or paired, and why those algebraic properties matter for diagnosing progress.

To move from description to diagnosis, we classify transformations by their algebraic properties. Most of these maps are parameterized, so claims such as idempotence, commutation, and adjunction are stated up to natural isomorphism and become strict only in idealized or limiting regimes. In discrete and symbolic settings, rewrite systems and search operators play the role of transformations, so non-commutativity can be a design choice rather than a defect.

### 5.1 Idempotent Functors

In plain language, some transformations stabilize after one application, like abstraction or projection, while others only behave this way in special cases such as L1 sparsity, capacity control, or augmentation.

Abstraction and quotient-like operations are idempotent: once applied, reapplication yields no further change. This identifies *stabilization points* in research pipelines. A functor $F$ is idempotent if $F\circ F \cong F$.

Strict or canonical cases:

* **Abstraction / quotient reflectors.** If $Q:\mathcal{C}\to\mathcal{A}$ is a left adjoint to an inclusion $U:\mathcal{A}\hookrightarrow\mathcal{C}$, then $R = U\circ Q$ is a reflector and $R\circ R \cong R$.
* **Projection to constrained subspaces.** True projections (top-$k$ pruning with deterministic tie-breaks, rank-$k$ truncation) are idempotent and can be modeled as reflectors onto a subcategory.

Idempotent in effect (common in ML, not exact):

* **Sparsity induction** via $L_1$ penalties is an update map, not a projection; idempotent only when it converges to a fixed projection.
* **Capacity control** via early stopping is not idempotent, whereas strict pruning is.
* **Data augmentation** is idempotent if it closes a dataset under a group action, but only "in distribution" when augmentation is stochastic.

### 5.2 Commutation and Non-Commutation

In plain language, order usually matters: regularization vs scaling and augmentation vs regularization may commute only after retuning, while abstraction vs optimization, bias vs manifold fitting, sparsity vs auxiliary losses, and capacity vs scaling are strongly non-commuting.

Most generative transformations do not commute. Order matters, and this is not a nuisance but a source of information: non-commutativity exposes where architectural choices constrain optimization, or where sparsity interacts destructively with auxiliary objectives. In logic, Bayesian inference, or evolutionary search, non-commutativity can be a deliberate way to preserve guarantees or exploration pressure. When commutation does hold, it is often only after a parameter transport, captured as a natural transformation $F\circ G \Rightarrow G\circ F$.

Frequently commuting (up to reparameterization):

* **Regularization and scaling** commute if regularization strength is rescaled with width, batch size, or learning rate.
* **Augmentation and regularization** often commute up to loss reweighting.
* **Auxiliary objectives** commute when losses are additive and share a representation, but not when they reshape architecture or sampling.

Strongly non-commuting (order matters in practice and concept):

* **Abstraction vs. optimization shaping** changes the dynamics themselves.
* **Inductive bias injection vs. manifold fitting** can redefine the effective geometry.
* **Sparsity vs. auxiliary objectives** can destroy pretext signals if applied too early.
* **Capacity control vs. scaling** differs except in narrow regimes (e.g., lottery ticket assumptions).

### 5.3 Adjunctions as Design Signals

In plain language, adjunctions capture clean paired moves such as abstraction versus inclusion, augmentation versus invariants, and structure injection versus forgetting when a universal property exists.

Adjunctions identify paradigms where added structure is both universal and legible. Paradigms lacking such adjunctions tend to resist systematic extension.

Clean adjunctions you can state without overpromising:

* **Abstraction / concretization.** $Q \dashv U$ for abstraction and inclusion; $U\circ Q$ is idempotent and gives the best approximation in the abstract subcategory.
* **Augmentation / invariants (group actions).** Orbit-closure behaves like a free construction (left adjoint) to taking invariants or quotients, when modeled as $\mathcal{G}$-objects.
* **Structure injection / forgetting.** Many inductive biases admit a free construction left adjoint to a forgetful functor (e.g., equivariant lifts), though this holds only when a universal property exists.

---

## 6. A Minimal Generating Set of Research Operations

In plain language, this section lists the generator types (P, L, D, O, S, Q, I, R), explains scaling cost models, and notes the set is minimal only relative to the modeling scope.

A minimal basis for comparing connectionist, symbolic, Bayesian, and evolutionary paradigms expands the generator set to cover inference and theory revision:

$$\mathbb{G}^* = \{ \mathbf{P}, \mathbf{L}, \mathbf{D}, \mathbf{O}, \mathbf{S}, \mathbf{Q}, \mathbf{I}, \mathbf{R} \}$$

Where:

* $\mathbf{P}$: parameterization / architecture constructors
* $\mathbf{L}$: objective / loss constructors
* $\mathbf{D}$: data and distribution constructors
* $\mathbf{O}$: optimization and training dynamics constructors
* $\mathbf{S}$: scaling constructors, parameterized by cost model (e.g., $\mathbf{S}^{(\mathrm{LA})}$, $\mathbf{S}^{(\mathrm{Search})}$, $\mathbf{S}^{(\mathrm{Inference})}$, $\mathbf{S}^{(\mathrm{Population})}$)
* $\mathbf{Q}$: abstraction / quotient constructors
* $\mathbf{I}$: inference / reasoning constructors (exact vs approximate inference, message passing, proof strategies)
* $\mathbf{R}$: rule / program / theory revision constructors (rule editing, macro invention, DSL evolution)

All standard techniques are compositions of these generators, but different paradigms activate different subsets.

Minimality here is relative to the chosen modeling of transformations and the scope of paradigms considered.

Derived families (examples):

* **Regularization** as $\mathbf{L}$, **auxiliary objectives** as $\mathbf{L}$, **inductive bias injection** as $\mathbf{P}$.
* **Data augmentation** as $\mathbf{D}$, **optimization shaping** as $\mathbf{O}$, **scaling** as $\mathbf{S}$, **abstraction** as $\mathbf{Q}$.
* **Inference upgrades** as $\mathbf{I}$ (exact to approximate, new kernels or message schedules).
* **Theory or program revision** as $\mathbf{R}$ (macro libraries, rule induction, proof tactics).
* **Manifold fitting** as $\mathbf{P}\circ\mathbf{D}\circ\mathbf{L}$; **sparsity/capacity** as projections (idempotent) or as $\mathbf{P}$ with $\mathbf{L}$ and $\mathbf{O}$ when implemented via penalties and dynamics.

---

## 7. Case Study: Neural Networks vs. Kernel Machines

In plain language, this section compares neural networks and kernels under shared validity and quality definitions: neural networks cover more generators and scale under linear-algebra cost models but have weaker canonical abstraction and translation back to symbols, while kernels keep stronger inference guarantees with narrower parameter choices and scaling limits.

This comparison is illustrative rather than definitive. It conditions on shared $\mathcal{Q}$ and $\mathrm{Valid}$ choices and on the dominant cost model used in practice.

Neural networks and kernel machines are both expressive, but they activate different parts of $\mathbb{G}^*$ and scale under different cost models.

Neural networks:

* **Generator coverage**: strong $\mathbf{P}, \mathbf{L}, \mathbf{D}, \mathbf{O}$; strong $\mathbf{S}^{(\mathrm{LA})}$; weaker $\mathbf{Q}, \mathbf{I}, \mathbf{R}$ as first-class operators.
* **Cross-fiber translations**: relaxation and distillation are common, but canonical extraction of symbolic structure remains weak.
* **Validity**: improvements are often empirical rather than guaranteed, so $\mathrm{Valid}$ constraints can be fragile.

Kernel machines:

* **Generator coverage**: strong $\mathbf{L}$ and $\mathbf{O}$ via convex training, strong $\mathbf{I}$ through exact or well-characterized inference; $\mathbf{P}$ is narrow (kernel choice), $\mathbf{R}$ is minimal.
* **Scaling**: $\mathbf{S}^{(\mathrm{LA})}$ is constrained by kernel matrix costs; approximations change the semantics.
* **Validity**: guarantees are strong within their semantic commitments, but the reachable set is smaller because fewer generators compose deeply.

On this view, neural networks are not universally superior; they are fertile under a scaling-friendly cost model and broad generator coverage. Kernel methods are fertile within a tighter semantic and validity regime, but have fewer composable paths and weaker cross-fiber translations, which limits their long-term reachability.

---

## 8. Applying the Diagnostic to Neurosymbolic Paradigms

In plain language, this section scores neurosymbolic families using the expanded generators and human channels to identify hubs and missing pieces, conditional on explicit validity and cost assumptions.

We treat "neurosymbolic" as a family of paradigms, then score each family along the expanded generators $\mathbb{G}^* = \{\mathbf{P}, \mathbf{L}, \mathbf{D}, \mathbf{O}, \mathbf{S}, \mathbf{Q}, \mathbf{I}, \mathbf{R}\}$ and the human channels $(\mathrm{Spec}, \mathrm{Realize}, \mathrm{Explain})$. The goal is to identify paradigm-hubs versus endpoints and to diagnose what is missing for higher generative potential, conditional on explicit validity criteria and cost models.

### 8.1 The checklist

In plain language, the checklist asks about generator coverage, validity preservation, cross-fiber translations, human channels, and scaling, and it is qualitative and depends on fixed definitions.

The checklist is qualitative and assumes explicit choices of $\mathbb{F}$, $\mathcal{Q}$, $\mathrm{Valid}$, and cost model.

A neurosymbolic candidate is promising if it has:

* **Generator coverage**: multiple generators in $\mathbb{G}^*$ are first-class and composable, not collapsed into a single choice.
* **Validity preservation**: transformations stay within semantic constraints or return to them through explicit checks.
* **Cross-fiber translations**: principled mappings between neural and symbolic fibers (relaxation, extraction, summarization).
* **Human channels**: specification, realization, and interpretation are all supported.
* **Scaling mode**: a clear cost model for $\mathbf{S}$ (search, inference, linear algebra) with predictable returns.

### 8.2 Differentiable logic / soft unification (logic as a layer)

In plain language, differentiable logic systems combine neural learning with soft rules: they are strong in architecture, losses, and optimization, moderate in data and inference, weaker in canonical abstraction and revision, and scaling can erode logical fidelity. They translate symbolic-to-differentiable well but struggle to extract stable rules back, and human specification is good while explanations are mixed.

**Representative idea-state**: a neural model with a differentiable logic module (soft unification, fuzzy logic, differentiable forward chaining), trained end-to-end.

**Generator coverage**

* $\mathbf{P}, \mathbf{L}, \mathbf{O}$: strong (explicit symbolic module, logic losses, relaxed inference).
* $\mathbf{D}$: moderate (needs structured supervision and counterexamples).
* $\mathbf{I}$: moderate (soft inference, differentiable unification).
* $\mathbf{Q}, \mathbf{R}$: weak-to-moderate (rule extraction and revision are not canonical).
* $\mathbf{S}^{(\mathrm{LA})}$: weak-to-moderate (scaling can degrade symbolic fidelity unless constrained).

**Cross-fiber translations**

* Strong relaxations from symbolic to differentiable forms, but weak extraction back to stable symbolic objects.

**Human channels**

* $\mathrm{Spec}$: good (rules and constraints).
* $\mathrm{Realize}$: good (hybrid architectures).
* $\mathrm{Explain}$: mixed (soft logic is continuous but less inspectable).

**What is missing / what to improve**

* A canonical $\mathbf{Q}$ or $\mathbf{R}$ that turns learned soft rules into stable symbolic objects.
* Better commutation control between $\mathbf{O}$ and $\mathbf{P}$ so optimization does not exploit loopholes that violate semantic intent.
* A scaling regime $\mathbf{S}^{(\mathrm{LA})}$ that preserves logical validity.

**High-leverage direction**

Introduce an explicit abstraction and revision loop: learn in the soft space, project to discrete rules, and reinject. This creates a stable translation between neural and symbolic fibers.

### 8.3 Program induction / neural-guided program search

In plain language, program induction systems search programs with neural guidance and verification: neural scores translate into symbolic programs and symbolic verification can guide learning, they are strong in inference and search with clear human specification and explanations, but scaling is bottlenecked by combinatorics and verification cost.

**Representative idea-state**: a program space with search (enumeration, MCTS, constraint solving), guided by a neural scorer; a verifier checks correctness.

**Generator coverage**

* $\mathbf{P}, \mathbf{L}, \mathbf{D}, \mathbf{O}, \mathbf{I}$: strong (explicit program space, verification losses, search/inference).
* $\mathbf{Q}, \mathbf{R}$: strong potential (equivalence classes, sketches, macro invention).
* $\mathbf{S}^{(\mathrm{Search})}$: moderate (bottleneck is branching and verification).

**Cross-fiber translations**

* Strong: neural scoring translates into symbolic programs; symbolic verification can guide neural updates.

**Human channels**

* $\mathrm{Spec}$: very good (DSL design, constraints, invariants).
* $\mathrm{Realize}$: good (search procedures and verifiers).
* $\mathrm{Explain}$: excellent (programs are auditable).

**What is missing / what to improve**

* $\mathbf{S}^{(\mathrm{Search})}$ is brittle; scaling is dominated by combinatorics and verifier cost.
* Lack of a composable abstraction hierarchy $\mathbf{Q}$ with idempotent compression of programs into reusable macros.
* Weak commutation between $\mathbf{D}$ (task distribution) and program priors; systems overfit DSL idioms.

**High-leverage direction**

Make abstraction explicit and algebraic: treat sketches and macros as quotient objects, with idempotent compression into reusable higher-level constructs. This turns search into search in a learned abstraction lattice.

### 8.4 Constraint-based learning (hard constraints + neural components)

In plain language, constraint-based systems use explicit constraints with neural components: constraints make validity and interpretation strong, hard constraints can be relaxed into differentiable penalties but extraction is limited, abstraction and revision are weak, optimization interacts with constraints in fragile ways, and scaling constraints is difficult without modularization.

**Representative idea-state**: neural models trained subject to constraints (SAT/SMT, structured prediction constraints, or differentiable proxies).

**Generator coverage**

* $\mathbf{L}, \mathbf{O}$: strong (penalties, Lagrangians, constrained optimization).
* $\mathbf{I}$: moderate (constraint solving and feasibility checks).
* $\mathbf{P}, \mathbf{D}$: moderate (constraints often live outside the model).
* $\mathbf{Q}, \mathbf{R}$: weak (constraints do not automatically yield abstractions or revisions).
* $\mathbf{S}^{(\mathrm{Inference})}$: weak-to-moderate (constraints become harder with scale).

**Cross-fiber translations**

* Strong: hard constraints can be relaxed into differentiable penalties, but extraction of symbolic structure is limited.

**Human channels**

* $\mathrm{Spec}$: strong (constraints are direct priors).
* $\mathrm{Realize}$: good (constraint solvers and proxies).
* $\mathrm{Explain}$: strong (violations are interpretable).

**What is missing / what to improve**

* $\mathbf{Q}$ and $\mathbf{R}$: constraints regulate behavior but do not build a compositional library of concepts.
* Commutation fragility between $\mathbf{O}$ and $\mathbf{L}$; outcomes depend on schedules and multipliers.
* Scalable constraint management that decomposes global constraints into local checks.

**High-leverage direction**

Turn constraints into typed interfaces and local operators, not monolithic penalties. This shifts constraints from guardrails to generative connectors.

### 8.5 LLM tool-use agents (symbolic tools + neural controller)

In plain language, tool-using LLM agents are strong in controller design, data, and planning loops, but have weak abstraction and revision of skills; tool calls create cross-fiber moves but stable skill extraction is shallow, explanations and failure prediction are mixed, and training and prompting order still matter.

**Representative idea-state**: an LLM controller that calls external tools (search, solver, DB, theorem prover) with a protocol and memory.

**Generator coverage**

* $\mathbf{P}, \mathbf{D}, \mathbf{O}$: strong (controller, tool API, memory, planning loops).
* $\mathbf{L}$: moderate (alignment and tool-use losses are immature).
* $\mathbf{S}^{(\mathrm{LA})}$: strong (scales with model size and tool ecosystem).
* $\mathbf{I}$: moderate (planning and tool selection).
* $\mathbf{Q}, \mathbf{R}$: weak (skills exist but are rarely formalized or revised as objects).

**Cross-fiber translations**

* Tool calls are explicit translations to symbolic or procedural fibers, but extraction of stable programs or skills is weak.

**Human channels**

* $\mathrm{Spec}$: good (tools, constraints, prompts, policies).
* $\mathrm{Realize}$: good (tool protocols, memory design).
* $\mathrm{Explain}$: mixed (text is legible; internal policy is not).

**What is missing / what to improve**

* A formal $\mathbf{Q}$ and $\mathbf{R}$ that turn episodic behavior into stable skills with verifiable pre/post-conditions.
* Better failure-mode predictability via typed contracts and verifiers.
* Reduced non-commutativity between $\mathbf{O}$ (agent loop) and $\mathbf{L}$ (alignment/training); current systems are order- and prompt-sensitive.

**High-leverage direction**

Skill extraction as quotienting: learn policies, then compress to symbolic or typed plans with verifiable interfaces. This yields a genuine neurosymbolic library rather than a prompt zoo.

### 8.6 Cross-cutting diagnosis

In plain language, the common gaps are weak abstraction and revision, shallow translation loops between neural and symbolic forms, and scaling limits in verification and search.

Across the four families as typically implemented today, the following gaps recur under the assumptions above:

1. **Weak $\mathbf{Q}$ and $\mathbf{R}$**: symbolic structure is injected but rarely extracted or revised back out canonically, so reusable abstractions do not accumulate.
2. **Missing translation loops**: neural-to-symbolic and symbolic-to-neural translations are ad hoc rather than functorial, so cross-fiber reachability is shallow.
3. **Scaling is adversarial**: verification, search, and constraint propagation often scale worse than neural components, weakening the role of $\mathbf{S}$ as a connector.

### 8.7 Research agenda (framed in the diagnostic)

In plain language, the agenda is to make abstraction and revision explicit, build stable translation loops, and define scaling regimes for verification and abstraction under the right cost models.

If the goal is to select promising paradigms and identify improvement directions, three high-yield bets emerge:

1. **Make $\mathbf{Q}$ and $\mathbf{R}$ explicit, idempotent, and composable** across rules, programs, skills, and constraints.
2. **Engineer translation loops** so specification and interpretation are paired with stable extraction and reinjection of structure.
3. **Define scaling regimes for verification and abstraction** by treating them as optimizers $\mathbf{O}$ that must scale smoothly under their native $\mathbf{S}$ via modularity, locality, caching, and learned heuristics with guarantees.

---

## 9. Paradigm Selection and Diagnosis

In plain language, this section turns the framework into practical questions about generator coverage, validity, translations, human channels, and scaling.

The framework is intended as a **tool for future research**, not post hoc explanation.

Given a candidate paradigm, one should ask:

1. **Generator coverage in $\mathbb{G}^*$**
   Which of $\mathbf{P}, \mathbf{L}, \mathbf{D}, \mathbf{O}, \mathbf{S}, \mathbf{Q}, \mathbf{I}, \mathbf{R}$ are first-class and composable?

2. **Validity preservation**
   Do transformations stay within $\mathrm{Valid}$, or is there a stable path back to validity?

3. **Cross-fiber translations**
   Are there principled $T_{s\to s'}$ maps between semantic commitments, or is the paradigm isolated?

4. **Human channels**
   Are specification, realization, and interpretation supported with stable interfaces?

5. **Scaling mode**
   Is scaling defined under the paradigm's native cost model, with predictable returns?

This reframes "what is lacking?" into a precise question about missing generators, missing translations, or broken validity and human channels.

---

## 10. Direction of Study

In plain language, this section recommends designing paradigms with clear interfaces, explicit abstraction and revision, first-class inference, and scaling regimes that match the cost model.

The proposed direction is not to search blindly for new models, but to:

* design paradigms with explicit internal interfaces and cross-fiber translations,
* expose abstraction and revision operators ($\mathbf{Q}$ and $\mathbf{R}$),
* treat inference as first-class ($\mathbf{I}$) rather than hiding it in optimization,
* and define scaling regimes appropriate to each cost model.

Progress, in this view, comes from **engineering the idea space** to be navigable, rather than from isolated improvements in performance.

---

## 11. Conclusion

In plain language, the conclusion says fertile paradigms expand reachable valid ideas through composable generators and translation loops, and the goal is reliable expansion rather than copying one paradigm.

We argue that the most promising research paradigms are those with high generative potential: paradigms that support composable generators across implementation, inference, and theory revision; preserve semantic validity; and provide bidirectional translations between formal artifacts and human-usable structure. The framework presented here offers a way to identify such paradigms early, diagnose stagnation, and guide the design of future systems.

The lesson is to build systems that preserve validity while expanding the space of reachable, interpretable ideas.

## 12. Residual Uncertainty

In plain language, this section notes limits that remain: the framework depends on chosen scoring and validity definitions, cross-fiber translations may not exist, and scaling advantages can shift with hardware.

* The framework depends on the choice of $\mathcal{Q}$, $\mathrm{Valid}$, and $\mu$, and different choices can reverse comparative conclusions.
* Cross-fiber translations are assumed to exist and be principled; many domains lack such mappings in practice.
* Generator coverage and composability are modeled abstractions; real systems may blur or collapse generators in ways the model does not capture.
* Cost models for scaling change with hardware and infrastructure, which can shift which paradigms appear fertile.