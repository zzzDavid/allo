<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# One complete histogram confirmation row

Registered in round209 before any new histogram execution, under the user's
explicit authorization. The failed round208 expert control and the entire
incomplete row remain retained. No individual sample will be replaced and
no samples from that row enter this new row's statistic.

After the owed native-kernel campaign and the residual-space go/no-go gate,
run exactly one fresh complete paired row in a new absolute evidence directory:

```sh
python -m benchmarks.apu_v1.paper_rows --measured 10 --expert \
  --historical-binary --cases histogram --scratch "$H/scratch" --out "$H/row.json"
```

Use the named installed artifact and copied benchmark tree, unchanged
evaluation fixtures, source/shared selection, arithmetic, three lanes,
one warm-up per lane and ten interleaved measured rounds (33 attempts).
Retain the device-host lock, exact numerical checks, exactly-one-fresh-record
requirement, all raw logs, generated source/binary identities and failures.
Provision host-local TMPDIR/profile/evidence storage and run the actual
residual allocation/storage/transport preflight before starting the driver.

The row is complete only when every prescribed launch is valid. Compare
the ten measured Tenon samples using the existing median and paired baseline
rule, against the frozen paper and fresh expert/historical lanes. Report
quartiles/raw samples; no new tolerance. Any missing/invalid launch leaves
the row INCOMPLETE, with no automatic replacement row or sample. This one-row
authorization is separate from the native campaign's remaining four-attempt
ceiling and does not replenish that ceiling.

Follow with the installed G6 board checks, retaining the V1-16
`raises=BitModelMismatch` boundary. A transport failure is a failing test,
never evidence for that expected arithmetic-model mismatch.
