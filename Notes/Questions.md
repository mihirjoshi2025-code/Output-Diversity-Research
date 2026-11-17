- [ ] How are the percentages of Core stories from each fine-tuning checkpoint calculated in the paper?
	- [ ] Does the DPO figure analyze the number of Core stories within the accepted dataset?
	- [ ] Count only real stories? Do the percentages from the paper look at all responses? If so, should we filter down to just stories as that's what we're prompting on?
- [ ] Analyze teacher data: Score stories from DeepSeek R1, Qwen3-32B, and Qwen3-0.6B directly which could show habits OLMo might've picked up.
- [ ] Should we study the base models outputs as well to see if the model generates Core stories before post training?

pick instruct over think


# Development and Validation Record

  

## Milestone 1: Cash-only run on a tiny market file

  

**Initial plan.** Running the executable on `data/handout.csv` with the `cash`

strategy and $100.00 reads and validates the market file, steps through each

date, and writes an output CSV with a `date,value` header and one row per

market date. Every value equals the starting cash. Planned tests:

- Command-line run  with the market file `data/handout.csv`, output out.csv, cash 100.00 and strategy

`cash`. Then cat out.csv. Expected output, worked out by hand:

date,value

2024-01-02,100.00

2024-01-03,100.00

2024-01-04,100.00


- OUnit: load `data/handout.csv` through the market module and check that it

has the securities `ALFA` and `BETA` and three dates in file order.


**Actual result.** Not yet performed.

## Milestone 2: Invalid market files are rejected with a clear error

**Initial plan.** If a market file breaks any R1 format rule, the executable
prints an error, exits with a nonzero status, and writes no output file. Valid

files still run as in Milestone 1. Planned test: tiny invalid files that each

break exactly one rule, each run with the cash strategy:

- header that doesn't start with `date`, or has no security columns

- duplicate security name

- a row with too few or too many fields, and a blank line

- bad dates (`2024-02-30`, `2024-1-05`) and dates that aren't strictly

increasing

- prices `0.00`, `10.0`, `.50`, `10.000`, ` 10.00`, `"10.00"`

- a header with no data rows

  

Expected: each one gives an error and no output file. An OUnit version checks

that the market loader rejects each file and accepts `data/handout.csv`.

  

**Actual result.** Not yet performed.

  

## Milestone 3: Buy-and-hold run with correct portfolio accounting

  

**Initial plan.** Running with the buy-and-hold strategy makes its purchase on

the first date at that day's closing prices and then holds. Each output value

is the remaining cash plus the holdings valued at that date's closing prices.

It runs through the same engine as `cash`. Planned tests:

- Before running, work out by hand on `data/handout.csv` the cash, holdings

and value after each date (recorded under *Buy-and-hold predictions*), then

compare with the program's output.

- OUnit on the portfolio: on a small multi-date example, apply a rejected

order (e.g. buying more than the cash allows) and then a valid one. Check

that the rejected order changes nothing and that cash, holdings and value

after the valid one match my hand calculation.

  

**Actual result.** Not yet performed.

  

## Milestone 4: Stateful strategy behind the shared strategy interface

  

**Initial plan.** Running with the stateful strategy produces a full

`date,value` report. The strategy's state is passed from day to day as a value.

`cash`, buy-and-hold and the stateful strategy share one strategy signature,

and the engine has no branch for any particular strategy's rule. Planned tests:

- On a new tiny file built so the rule fires on certain days, write down the

orders I expect on each date and the resulting values *before* writing the

test. Then an OUnit test compares the engine's dated values with that

prediction.

- Searching the engine source for strategy names finds nothing.

  

**Actual result.** Not yet performed.

  

## Milestone 5: Complete command-line path and required test groups

  

**Initial plan.** All three strategies can be selected from the command line

(BatOptParse). An unknown strategy, missing or nonnumeric cash, or a missing

market file produces an error or usage message and a nonzero exit. Planned

tests:

- Cram runs of each strategy with $10,000.00 on `data/small-spy-5days.csv`,

checking the complete standard output against values calculated separately

from the program.

- Cram runs for each command-line error, checking the message and exit code.

- A new valid market file with different security names, run through all

three strategies.

- `dune build` and `dune test` from the project root, both clean.

  

**Actual result.** Not yet performed.

  

## Cash on the supplied tiny market file

  

With $100.00 on `data/handout.csv`, the cash strategy never trades. On every

date (2024-01-02, 2024-01-03, 2024-01-04), cash is $100.00, holdings are empty,

and the portfolio value is $100.00. OUnit group (planned name):

`cash on handout`, which checks every date.

  

## Buy-and-hold predictions

  

To be completed once buy-and-hold is implemented (Milestone 3): the initial

cash and held security for both market files, the new tiny multi-date,

multi-security file and why it was chosen, and the first purchases and dated

results checked by the buy-and-hold OUnit group.

  

## Stateful-strategy predictions

  

To be completed during Milestone 4: whether I used the default momentum rule

or my own (and if my own, a precise description), the new tiny file and why it

was chosen, the predicted behavior on both market files, and the OUnit group

that checks it.

  

## Command-line report check

  

To be completed during Milestone 5: the expected final results for all three

strategies with $10,000.00 on `data/small-spy-5days.csv`, and the Cram group

that checks each run's complete standard output.

  

## Consequential revisions

  

None yet. Any change to milestone order, scope or tests will be recorded here

along with how the result was tested.