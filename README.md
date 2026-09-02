QUATERNION SECURITY-ATTRIBUTION BENCHMARK
=========================================

1. PURPOSE

This repository provides executable simulations for security attribution in
quaternion image-protection pipelines. It evaluates the protected image body,
the complete serialized public object, permitted functionality leakage,
active modification behavior, forced-reuse relations, public inversion,
structured recovery, learned reconstruction, distinguishing behavior, and
systems cost.

The benchmark contains 24 registered constructions. The set includes standard
authenticated-encryption controls, quaternion and transform pipelines,
publicly invertible controls, deterministic controls, freshness controls, and
fixed-header controls. Several constructions are deliberately insecure or
publicly reversible. They are calibration controls and must not be used as
production cryptography.

2. REPOSITORY CONTENTS

  src/qsa_benchmark/benchmark
      Construction registry, QSB1 envelope format, quaternion transforms,
      cryptographic controls, corpus generation, instrumentation, and the
      compact benchmark runner.

  src/qsa_benchmark/protocol
      Differential protocols, leakage analysis, attack experiments, timing
      decomposition, replication checks, primitive scaling, and targeted
      validation.

  src/qsa_benchmark/security
      Exact finite-domain identities, counterexamples, attribution checks,
      and algebraic recovery routines.

  src/qsa_benchmark/attacks
      Permutation recovery, affine-diffusion recovery, and reconstruction
      metrics.

  configs/benchmark
      A fast 20-construction smoke configuration and a larger 20-construction
      core diagnostic configuration.

  configs/protocol
      The authoritative 24-construction experiment, primitive-scaling
      protocol, and targeted-validation protocol with JSON schemas.

  configs/reference
      Fixed public input, key, nonce, seed, context, and active-modification
      definitions with validation schemas.

  configs/verification
      The unified repository-verification contract and schema.

  registries
      Machine-readable construction, corpus, leakage, perturbation, protocol,
      and active-modification registries.

  reference/known_answers
      One canonical RGB input, a machine-readable manifest, and 24 frozen
      complete-object reference files covering B01 through B24.

  reference/malformed_objects
      A machine-readable manifest and 33 frozen QSB1 parser-rejection vectors.

  reference/deterministic_outputs
      The 37-artifact inventory, frozen comparison hashes, and protocol-binding
      digest for 13 semantic CSV tables and 24 serialized objects.

  reference/environment
      The validated interpreter, platform, package, metadata, and numerical
      runtime record associated with requirements-lock.txt.

  reference/verification
      The frozen compact-benchmark fingerprint and corpus-manifest identity.

  src/qsa_benchmark/validation
      Environment, reference-asset, registry, neutrality, and unified
      verification utilities.

  scripts
      Stable commands for data generation, verification, protocol execution,
      explicit reference management, and optional native-accelerator building.

  tests
      Registry, environment, fixed known-answer, malformed-object,
      active-modification, deterministic-artifact, cross-execution,
      primitive-scaling, round-trip, integrity, finite-domain, and targeted
      validation checks.

3. REQUIREMENTS

Python 3.11 or newer is required. The exact validated environment uses CPython
3.13.5 and the package versions recorded in requirements-lock.txt. The exact
environment manifest also records package METADATA digests, the operating
platform, the OpenSSL runtime, and the NumPy numerical-runtime configuration.

The optional native accelerator requires a C compiler, OpenMP support, and
OpenSSL development libraries. The Python reference implementation remains
available when the accelerator is absent.

4. INSTALLATION

For an ordinary installation from the repository root, run

  python -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e ".[test]"

On Windows PowerShell, activate the environment with

  .venv\Scripts\Activate.ps1

For the exact validated package set, create a clean environment and run

  python -m pip install --upgrade "pip==25.1.1"
  python -m pip install -r requirements-lock.txt
  python -m pip install --no-deps --no-build-isolation -e .
  python scripts/lock_environment.py --check --policy exact

The exact policy intentionally fails when the interpreter, platform, package
versions, installed package metadata, OpenSSL runtime, or NumPy numerical
runtime differ from the validated environment. A package-only comparison is
available with

  python scripts/lock_environment.py --check --policy packages

The report policy records differences without rejecting the execution.

5. UNIFIED VERIFICATION

The authoritative repository-verification command never changes frozen
references.

Static verification checks version and license consistency, the exact
software environment, public neutrality, prospective archive safety, the
SHA-256 inventory, JSON schemas, parent-protocol linkage, registry parity, and
registered counts.

  python scripts/verify_repository.py --static

Default verification first regenerates every machine-readable registry in a
temporary workspace and verifies corpus, perturbation, construction, leakage,
and protocol parity. It then adds the complete automated test suite, all 24
fixed known-answer cases, all 33 malformed-object vectors, the 180-case active-
modification schedule, the frozen 37-artifact contract, and the compact
20-construction fingerprint.

  python scripts/verify_repository.py

Full deterministic verification creates clean primary and independently
initialized execution trees. It verifies the 13 deterministic tables and 24
representative serialized objects against the frozen reference and across the
two trees. It also executes targeted validation and 48 primitive round-trip
cases per execution, giving 96 exact cases in total.

  python scripts/verify_repository.py --full

Platform-conditioned timing verification additionally evaluates all 576
registered complete-path timing configurations and the primitive-scaling
timing records.

  python scripts/verify_repository.py --full --timing

Each tier writes a JSON summary and a text summary under
results/verification/<tier>. Passing commands return status 0. Static and
neutrality failures return 2. Environment failures return 3. Computational,
count, fixed-answer, and expected-outcome failures return 4. Timing-limit
failures return 5.

Deterministic summaries contain no wall-clock timestamps or absolute paths.
Timing summaries record platform metadata and remain outside deterministic
artifact equality.

6. REFERENCE ENVIRONMENT MANAGEMENT

Verify the committed environment files without changing them with

  python scripts/lock_environment.py --check --policy exact

Environment replacement is explicit.

  python scripts/lock_environment.py --update --policy exact

The update command regenerates requirements-lock.txt and
reference/environment/validated_environment.json from the active environment,
refreshes SHA256SUMS.txt, and verifies the resulting lock.

7. FAST COMPONENT CHECKS

Run the complete automated test set with

  python -m pytest -q

Verify all 24 frozen known-answer cases without changing them with

  python scripts/generate_reference_assets.py known-answers --check

Verify the malformed-object set without changing it with

  python scripts/generate_reference_assets.py malformed-objects --check

Verify the deterministic-artifact inventory, frozen hashes, protocol binding,
and recorded cross-execution freeze without running the full experiment with

  python scripts/generate_reference_assets.py deterministic-artifacts --check

Verify that every committed registry matches its executable authority with

  python scripts/export_registries.py --check

Run the compact 20-construction benchmark with

  python scripts/run_smoke.py

8. FIXED KNOWN-ANSWER REFERENCE SET

The reference set uses one deterministic 16-by-16 RGB input and one public,
domain-separated context for every registered construction. Verification
regenerates each complete serialized object, compares its exact bytes and
component SHA-256 digests with the frozen answer, and then checks exact input
recovery. This is stronger than a round-trip check alone because a matching
forward and inverse defect cannot silently replace the frozen-output test.

All keys, nonces, seeds, and public materials in this directory are published
test values. They must not be used for operational protection. Reference
replacement is intentionally explicit.

  python scripts/generate_reference_assets.py known-answers --update

9. MALFORMED OBJECTS AND ACTIVE MODIFICATIONS

The 33 malformed-object vectors exercise the canonical QSB1 parser
independently of construction-specific release behavior. Every frozen vector
is rejected with one of 15 stable EnvelopeFormatError codes covering prefix,
header, canonical encoding, field semantics, component length, or object
framing. Explicit replacement is performed only with

  python scripts/generate_reference_assets.py malformed-objects --update

The active-modification registry contains 180 byte-distinct, method-specific
probes over all 24 constructions. It separates 48 parser-level append and
truncation cases from construction-level release behavior. All 73
authenticated cases are rejected. The 107 unauthenticated cases contain 30
parser-level length rejections and 77 accepted non-length modifications. The
accepted cases divide into 37 changed recovered images and 40 unchanged
recovered images. Public-payload splice probes are used for B05 through B11 so
every registered modification is byte-distinct and operational.

10. DATA GENERATION

Generate all registered image sizes with

  python scripts/prepare_data.py

Generate selected sizes with

  python scripts/prepare_data.py --sizes 96 256 512

Synthetic probes are generated analytically. Natural samples are obtained from
the scikit-image data module and resized deterministically. Each generated file
is recorded with its source label, dimensions, semantic category, license
note, raw-pixel digest, and encoded-file digest.

11. COMPACT BENCHMARK COMMANDS

Validate the smoke configuration with

  qsa-benchmark validate-config --config configs/benchmark/smoke.yaml

Generate its corpus with

  qsa-benchmark prepare-data --config configs/benchmark/smoke.yaml

Execute it with

  qsa-benchmark run --config configs/benchmark/smoke.yaml

Regenerate the corpus and replace an existing output directory with

  qsa-benchmark reproduce --config configs/benchmark/smoke.yaml

The larger 20-construction diagnostic uses

  configs/benchmark/core_full.yaml

Neither compact configuration replaces the authoritative 24-construction
protocol under configs/protocol/experiment.json.

12. FULL 24-CONSTRUCTION PROTOCOL

Execute two separately initialized full executions with

  python scripts/run_experiment.py

The command writes the two execution trees under results/experiment and checks
all 37 deterministic artifacts against each other and against the frozen
reference manifest. Every table is checked for its registered row count,
ordered column schema, semantic digest, exact raw serialization, and byte size.
Every representative object is checked byte-for-byte, parsed as QSB1, and
reconciled with its manifest row. The command also evaluates cross-execution
timing limits and cost-model sample counts. Absolute latency remains
platform-conditioned.

To regenerate and verify only deterministic artifacts, omit timing with

  python scripts/run_experiment.py --skip-timing

Reference replacement requires two complete deterministic trees.

  python scripts/generate_reference_assets.py deterministic-artifacts --update --results-root results/experiment

13. PRIMITIVE-SCALING PROTOCOL

The primitive-scaling protocol evaluates two protection primitives, four
registered images, three image sizes, and two body representations. The
factorization gives 48 exact round-trip cases per execution and 96 cases across
two independently initialized executions.

Execute exact recovery without timing with

  python scripts/run_primitive_scaling.py

Include the registered primitive timing measurements with

  python scripts/run_primitive_scaling.py --timing

The timed region contains primitive protection or unprotection only. Transform
computation and complete-object serialization are excluded. The raw RGB body
uses one byte per coordinate. The Case-II body uses signed little-endian int32
coordinates and therefore has an exact body-length ratio of four.

14. TARGETED VALIDATION

Execute the B13 permutation-query experiment and the B23 fixed-header power
calibration in two independently initialized runs with

  python scripts/run_targeted_validation.py

The B13 experiment verifies the exact base-256 query threshold for identifying
a byte-position permutation. The B23 experiment calibrates NPCR rejection
power when a fixed 32-byte prefix is combined with an idealized independent
suffix model.

15. OUTPUT INTERPRETATION

NPCR and UACI are reported for explicitly identified projections and operation
regimes. A strong protected-body statistic is not treated as evidence about
unexposed object fields. Public metadata, nonces, previews, descriptors,
protected payloads, and tags are analyzed according to the complete public
object model.

The attack-leakage-regime matrix is the primary compact endpoint. It reports
public inversion, structured recovery, learned reconstruction, protected-body
distinguishing, active modification, forced-reuse relations, permitted preview
leakage, and the correct-use confidentiality class separately.

16. COMPUTATIONAL REPRODUCIBILITY

Computational reproducibility concerns deterministic artifacts generated from
the released implementation, registries, schedules, and locked environment.
Cross-execution agreement concerns equality of deterministic artifacts
generated by two separately initialized execution trees.

The committed deterministic layer contains an executable inventory, a
protocol-binding digest, and frozen identities for 13 CSV tables and 24
representative QSB1 objects. The primary table digest is computed from
canonical string-valued rows. The verifier also enforces row counts, ordered
column schemas, raw-file digests, line endings, final newlines, and byte sizes.
Each representative object is compared by exact bytes and parsed through the
canonical QSB1 parser.

Timing results are platform-conditioned. Two local execution trees are
compared through registered relative-difference limits rather than literal
timing equality. The second execution verifies implementation and schedule
stability. It does not enlarge the statistical sample.

17. SCOPE

This code is a research benchmark. It is not a deployable encryption library.
Standard cryptographic primitives are used as controls through maintained
Python libraries. Transform-only, public, deterministic, and deliberately
misused constructions exist solely to expose attribution errors and calibrate
measurements.

