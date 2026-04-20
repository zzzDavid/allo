"""Runtime/execution support for each target.

- aim_shadow: Python ISR-level functional model of SK-Hynix AiM. The real
  aim_simulator (Ramulator 2.0) is trace/timing only; the shadow exists so
  we can verify numerical correctness of the ISRs we emit.
- pimsim_driver: Python wrapper that invokes the Samsung PIMSimulator C++
  driver (pim_drivers/samsung/pim_driver) with a kernel spec.
- upmem_codegen: DPU task.c + host driver + assembler glue for uPIMulator.
"""
