..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

####################################
Processing-in-Memory (Tenon) Targets
####################################

The ``tenon`` branch of Allo adds five processing-in-memory (PIM) targets
behind one compile entry point.  A workload written once against the Allo
dataflow surface is compiled for a target description and a cost program,
and the compiler chooses the layout, the placement and the schedule by cost::

    import allo
    from allo.pim import build_samsung_target, samsung_cost

    compiled = allo.compile(workload, build_samsung_target(), samsung_cost)
    result = compiled(W, x, y)          # RunResult: kind, cycles, extra
    estimate = compiled.estimate()      # cost of the selected program

``allo.compile`` returns one ``CompiledCallable`` for every target.  Calling
it with NumPy arrays runs the selected program on the target's executor and
updates the output arrays in place for executed runs; ``backend="virtual"``
evaluates the cost program only and leaves the arrays untouched.  The
architecture of the pipeline is described in ``SPMW_ARCHITECTURE.md`` at the
repository root; the release ledgers under ``docs/release/`` record what has
been validated and what has not.

Targets and executors
=====================

Target constructors live in ``allo.pim.targets`` and the cost programs in
``allo.pim.costs``; both are re-exported from ``allo.pim`` except
``attacc_cost``, which is imported from ``allo.pim.costs``.

.. list-table::
   :header-rows: 1
   :widths: 18 26 28 28

   * - Target
     - Constructor and cost
     - Executor
     - ``RunResult.kind``
   * - Samsung HBM-PIM
     - ``build_samsung_target()``, ``samsung_cost``
     - Samsung PIMSimulator through the Tenon conductor ``pim_driver``
       (``--op GENERIC``), built from the bundled recipe
     - ``simulator`` (functional, FP16 outputs written)
   * - SK hynix AiM
     - ``build_aim_target()``, ``aim_cost``
     - Ramulator 2.0 AiM model (timing only)
     - ``simulator-timing-only`` (no outputs written)
   * - UPMEM DRAM-PIM
     - ``build_upmem_target()``, ``upmem_cost``
     - uPIMulator through the UPMEM SDK (``GENERIC`` fixtures)
     - ``simulator`` (functional, integer outputs written)
   * - GSI Gemini-I (APU v1)
     - ``build_apu_v1_target()``, ``apu_v1_cost``
     - the board, through the GVML SDK; ``backend="functional"`` is a host
       model of the selected plan's arithmetic
     - ``device`` / ``functional``
   * - GSI Gemini-II (APU v2)
     - ``build_apu_g2_target()`` (alias ``build_apu_v2_target``), ``apu_g2_cost``
     - the card, through the GSI Gemini-II SDK
     - ``device``
   * - AttAcc (analytical)
     - ``build_attacc_target()``, ``attacc_cost``
     - none: ``backend="virtual"`` only; the Ramulator2-AttAcc figures are
       produced by ``benchmarks/attacc/scripts``
     - ``virtual``

Installation and discovery
==========================

The package is installed from source (``docs/release/INSTALL.md``,
sections 1 and 2).  Each executor is found at run time by
``allo.spmw_simenv``; when one is unavailable the compile still succeeds and
the run reports the reason instead of silently falling back.

Samsung PIMSimulator
    The simulator recipe is bundled with the package.  Build it once::

        python -m allo.pim.samsung_simulator build      # ~15 s on 16 cores
        python -m allo.pim.samsung_simulator locate

    The build lands under ``$TENON_HOME/simulators/PIMSimulator`` (default
    ``~/.local/share/tenon``) and is discovered from there.  ``PIMSIMULATOR_ROOT``
    and ``TENON_SIMULATORS_DIR`` keep precedence.  Read the bundled
    ``NOTICE.md``: the Samsung sources carry a non-commercial research licence.

SK hynix AiM
    ``AIM_SIMULATOR_ROOT`` (the Ramulator 2.0 checkout with the AiM model) and
    ``TENON_AIM_RAMULATOR2`` (a natively built ``ramulator2``); the AiM route
    is timing only and computes no data.

UPMEM
    ``TENON_UPMEM_SDK_HOME`` (the UPMEM SDK) and either the pinned campaign
    build or your own uPIMulator (``UPIMULATOR_ROOT``, ``UPIMULATOR_BIN``);
    see ``docs/release/INSTALL.md`` section 3.

GSI Gemini-I and Gemini-II
    The vendor SDKs and the devices; the Gemini-II SDK root and card index
    are read from ``TENON_APU_G2_SDK_ROOT`` and ``TENON_APU_G2_DEVICE``.
    Both routes take a per-device lock.

The full variable list and the simulator build instructions are in
``docs/release/INSTALL.md`` section 3 and ``docs/release/simulators.md``.

Inspecting a chosen program
===========================

``compiled.placement_ranking`` lists the candidates the compiler scored and
the one it selected, ``compiled.execution_graph`` is the retained graph the
estimate is computed from, and ``compiled.reduction_policy_pruned`` names the
candidates the reduction-order policy removed.  ``RunResult.extra`` carries
each executor's provenance (program and layout fingerprints, simulator
identities, device locks).  ``SPMW_ARCHITECTURE.md`` section 5 lists every
field.

What is validated
=================

``docs/release/CAPABILITIES.md`` states, per target, which workload families
compile and execute today and which are refused by name;
``docs/release/PAPER_REPRODUCTION.md`` and
``docs/release/paper_reproduction/inventory.csv`` record which published
results have been reproduced from installed artifacts and which remain open.
