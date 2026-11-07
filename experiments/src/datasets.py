"""Deterministic multi-step GSM8K & MATH datasets + tokenizer utilities for Qwen3 ablations.

Constructs paired reasoning problems across 6 arithmetic word-problem families (GSM8K)
and 6 algebraic / number-theoretic problem families (MATH), each containing:
1. `prompt`: Multi-step word problem (GSM8K) or algebraic equation/system (MATH).
2. `verbose_cot`: Full grammatical Chain-of-Thought trace ending in `#### <answer>`.
3. `telegraphic_cot`: Syntactically pruned state-transition trace via `prune_cot_trace(..., ratio=1.0)`.
4. `intermediate_values`: Stepwise numeric/symbolic targets for latent recurrence alignment.
5. `probe_stage_labels`: Symbolic operation classes supervising the detached `TorchSidecarProbe`.
6. `final_answer`: Exact canonical scalar answer string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from prune_syntax import prune_cot_trace

BenchmarkName = Literal["GSM8K", "MATH"]

PROBE_STAGE_TEMPLATES: tuple[str, ...] = (
    "[k={step_k}] bind prompt operands -> latent register",
    "[k={step_k}] op mul/div intermediate quantity",
    "[k={step_k}] op add/sub state accumulation",
    "[k={step_k}] algebraic factor / root / isolate step",
    "[k={step_k}] constraint check & branch filter",
    "[k={step_k}] halt_gate -> commit W_unembed answer",
)


@dataclass(frozen=True)
class ReasoningExample:
    """Single paired reasoning problem across all four GITLTYN inference regimes."""

    example_id: str
    benchmark: BenchmarkName
    prompt: str
    verbose_cot: str
    telegraphic_cot: str
    intermediate_values: tuple[str, ...]
    probe_stage_labels: tuple[int, ...]
    final_answer: str


def build_gsm8k_examples(n_problems: int, seed: int = 42) -> list[ReasoningExample]:
    """Generate deterministic multi-step GSM8K arithmetic word problems (2 to 5 hops)."""
    rng = np.random.default_rng(seed)
    names = ("Natalia", "Marcus", "Elena", "Devon", "Priya", "Lucas", "Hannah", "Omar")
    items = ("clips", "croissants", "notebooks", "orchids", "batteries", "tickets")
    examples: list[ReasoningExample] = []

    for idx in range(n_problems):
        person = names[idx % len(names)]
        obj = items[(idx // 2) % len(items)]
        pattern = idx % 6

        if pattern == 0:
            # 3-step rate + halving + June bonus
            base_even = int(rng.integers(12, 34)) * 2
            half_val = base_even // 2
            bonus = int(rng.integers(5, 19))
            sub_sum = base_even + half_val
            ans = sub_sum + bonus
            prompt = (
                f"{person} sold {base_even} {obj} in April, then sold half as many {obj} in May, "
                f"and finally sold {bonus} more {obj} in June. How many {obj} did {person} sell in total?"
            )
            verbose_body = (
                f"Let's think step by step to carefully calculate the total number of {obj} that {person} sold across all three months. "
                f"First, we know from the problem statement that in April {person} sold apr = {base_even} {obj}. "
                f"Next, because {person} sold half as many in May as in April, we divide by two so may = {base_even} / 2 = {half_val} {obj}. "
                f"Then, combining the April and May amounts together gives subtotal = {base_even} + {half_val} = {sub_sum} {obj}. "
                f"Therefore, adding the June sales gives the final total = {sub_sum} + {bonus} = {ans}."
            )
            inter = (str(base_even), str(half_val), str(sub_sum), str(ans))
            stages = (0, 1, 2, 5)
        elif pattern == 1:
            # 3-step Crates * per_crate with inspection subtraction
            per_box = int(rng.integers(8, 18))
            boxes_am = int(rng.integers(4, 11))
            boxes_pm = int(rng.integers(3, 9))
            kept = int(rng.integers(7, 19))
            total_boxes = boxes_am + boxes_pm
            gross = total_boxes * per_box
            ans = gross - kept
            prompt = (
                f"A workshop packs {per_box} {obj} into each crate. {person} prepares {boxes_am} crates "
                f"in the morning and {boxes_pm} crates in the afternoon, then sets aside {kept} {obj} "
                f"for quality inspection. How many {obj} are shipped?"
            )
            verbose_body = (
                f"Let's carefully determine how many {obj} are actually shipped from the workshop step by step. "
                f"First, we add the morning and afternoon crates together so crates = {boxes_am} + {boxes_pm} = {total_boxes}. "
                f"Next, since each crate holds {per_box} items, we multiply to get packed = {total_boxes} * {per_box} = {gross}. "
                f"Therefore, after we subtract the items kept aside for inspection, shipped = {gross} - {kept} = {ans}."
            )
            inter = (str(total_boxes), str(gross), str(kept), str(ans))
            stages = (0, 2, 1, 5)
        elif pattern == 2:
            # 4-step Dual-rate shifts minus recycled defects
            rate1 = int(rng.integers(9, 21))
            hrs1 = int(rng.integers(3, 6))
            rate2 = int(rng.integers(7, 17))
            hrs2 = int(rng.integers(2, 6))
            defect = int(rng.integers(5, 16))
            part1 = rate1 * hrs1
            part2 = rate2 * hrs2
            gross = part1 + part2
            ans = gross - defect
            prompt = (
                f"{person} processes {rate1} {obj} per hour for {hrs1} hours before lunch, "
                f"and then processes {rate2} {obj} per hour for {hrs2} hours after lunch. "
                f"At the end of the shift, {defect} damaged {obj} are discarded. "
                f"How many good {obj} remain?"
            )
            verbose_body = (
                f"Let's think step by step to find the number of undamaged {obj} remaining at the end of the shift. "
                f"First, during the morning shift before lunch, we calculate morning = {rate1} * {hrs1} = {part1}. "
                f"Next, during the afternoon shift after lunch, we compute afternoon = {rate2} * {hrs2} = {part2}. "
                f"Then, summing both shifts together yields gross = {part1} + {part2} = {gross}. "
                f"Therefore, subtracting the discarded damaged items leaves good = {gross} - {defect} = {ans}."
            )
            inter = (str(part1), str(part2), str(gross), str(ans))
            stages = (0, 1, 2, 5)
        elif pattern == 3:
            # 5-step Bundle + exact 20% coupon discount + flat shipping fee
            n_a = int(rng.integers(4, 10))
            price_a = int(rng.integers(3, 8)) * 5
            n_b = int(rng.integers(3, 8))
            price_b = int(rng.integers(2, 6)) * 5
            shipping = int(rng.integers(6, 15))
            cost_a = n_a * price_a
            cost_b = n_b * price_b
            subtotal = cost_a + cost_b
            discount = subtotal // 5
            discounted = subtotal - discount
            ans = discounted + shipping
            prompt = (
                f"{person} orders {n_a} packs of {obj} at ${price_a} each and {n_b} storage cases at ${price_b} each. "
                f"A store coupon takes 20% off the combined item subtotal, and flat shipping adds ${shipping}. "
                f"How many dollars does {person} pay in total?"
            )
            verbose_body = (
                f"Let's carefully compute the final checkout total in dollars step by step. "
                f"First, the packs cost pack_cost = {n_a} * {price_a} = {cost_a} and the cases cost case_cost = {n_b} * {price_b} = {cost_b}. "
                f"Next, combining both item groups gives subtotal = {cost_a} + {cost_b} = {subtotal}. "
                f"Then, taking the twenty percent coupon discount gives discount = {subtotal} / 5 = {discount}, leaving discounted = {subtotal} - {discount} = {discounted}. "
                f"Therefore, adding the flat shipping charge produces total = {discounted} + {shipping} = {ans}."
            )
            inter = (str(subtotal), str(discount), str(discounted), str(ans))
            stages = (0, 1, 4, 5)
        elif pattern == 4:
            # 4-step Three-day coupled cascade: d1, d2 = 2*d1 - drop, d3 = (d1 + d2)/2, total = d1 + d2 + d3
            d1 = int(rng.integers(7, 18)) * 2
            drop = int(rng.integers(2, 7)) * 2
            d2 = 2 * d1 - drop
            first_two = d1 + d2
            d3 = first_two // 2
            ans = first_two + d3
            prompt = (
                f"On Monday, {person} collected {d1} {obj}. On Tuesday, {person} collected {drop} fewer "
                f"than twice Monday's amount. On Wednesday, {person} collected half of the combined "
                f"Monday and Tuesday total. How many {obj} did {person} collect across all three days?"
            )
            verbose_body = (
                f"Let's track the amount collected on each day step by step before summing all three days together. "
                f"First, Tuesday's amount is twice Monday minus {drop}, so tue = 2 * {d1} - {drop} = {d2}. "
                f"Next, combining Monday and Tuesday together gives mon_tue = {d1} + {d2} = {first_two}. "
                f"Then, Wednesday is half of that combined two-day total, so wed = {first_two} / 2 = {d3}. "
                f"Therefore, summing all three days together yields total = {first_two} + {d3} = {ans}."
            )
            inter = (str(d2), str(first_two), str(d3), str(ans))
            stages = (0, 1, 2, 5)
        else:
            # 5-step Quotient + remainder gift-box revenue problem
            trays = int(rng.integers(4, 8))
            per_tray = int(rng.integers(9, 15))
            sample_used = int(rng.integers(3, 8))
            box_cap = int(rng.integers(6, 10))
            box_price = int(rng.integers(11, 19))
            baked = trays * per_tray
            usable = baked - sample_used
            full_boxes = usable // box_cap
            leftover = usable - full_boxes * box_cap
            ans = full_boxes * box_price + leftover * 2
            prompt = (
                f"{person} makes {trays} trays of {obj} with {per_tray} {obj} per tray, sets aside {sample_used} {obj} "
                f"as free samples, and packs as many full gift boxes of {box_cap} {obj} as possible. "
                f"Each full box sells for ${box_price} and each leftover {obj[:-1] if obj.endswith('s') else obj} sells for $2. "
                f"How many dollars does {person} earn?"
            )
            verbose_body = (
                f"Let's work through the production, packing, and revenue stages step by step. "
                f"First, total items produced is baked = {trays} * {per_tray} = {baked}, and after samples usable = {baked} - {sample_used} = {usable}. "
                f"Next, dividing by {box_cap} per box gives boxes = {usable} // {box_cap} = {full_boxes} full boxes with leftover = {usable} - {full_boxes} * {box_cap} = {leftover}. "
                f"Therefore, total earnings equal revenue = {full_boxes} * {box_price} + {leftover} * 2 = {ans}."
            )
            inter = (str(usable), str(full_boxes), str(leftover), str(ans))
            stages = (0, 1, 4, 5)

        pruned = prune_cot_trace(verbose_body, ratio=1.0).retained_text
        examples.append(
            ReasoningExample(
                example_id=f"gsm8k_{idx:04d}",
                benchmark="GSM8K",
                prompt=prompt,
                verbose_cot=f"{verbose_body} #### {ans}",
                telegraphic_cot=f"{pruned} #### {ans}",
                intermediate_values=inter,
                probe_stage_labels=stages,
                final_answer=str(ans),
            )
        )

    return examples


def build_math_examples(n_problems: int, seed: int = 108) -> list[ReasoningExample]:
    """Generate deterministic multi-step MATH algebra & number theory problems (3 to 5 hops)."""
    rng = np.random.default_rng(seed)
    examples: list[ReasoningExample] = []

    for idx in range(n_problems):
        pattern = idx % 6
        if pattern == 0:
            # Shifted perfect square: (n + k)^2 = r^2, n > 0
            shift = int(rng.integers(3, 11))
            ans = int(rng.integers(5, 19))
            root_val = ans + shift
            rhs = root_val * root_val
            b_coef = 2 * shift
            c_coef = shift * shift
            prompt = (
                f"Find the positive integer n such that n^2 + {b_coef}n + {c_coef} = {rhs}."
            )
            verbose_body = (
                f"Let's carefully determine the positive integer n that satisfies the quadratic equation n^2 + {b_coef}n + {c_coef} = {rhs}. "
                f"First, we recognize that the left side of the equation factors cleanly into the perfect square (n+{shift})^2 = {rhs}. "
                f"Next, taking the principal square root on both sides under the positive constraint gives n+{shift} = {root_val}. "
                f"Therefore, isolating n by subtracting {shift} from both sides produces n = {root_val} - {shift} = {ans}."
            )
            inter = (str(rhs), str(root_val), str(shift), str(ans))
            stages = (0, 3, 4, 5)
        elif pattern == 1:
            # Linear 2x2 system: x + y = S, x - y = D -> find x * y
            y_val = int(rng.integers(4, 15))
            diff_even = int(rng.integers(2, 9)) * 2
            x_val = y_val + diff_even
            s_val = x_val + y_val
            two_x = s_val + diff_even
            ans = x_val * y_val
            prompt = (
                f"Suppose integers x and y satisfy x + y = {s_val} and x - y = {diff_even}. "
                f"Compute the product xy."
            )
            verbose_body = (
                f"Let's solve this system of linear equations step by step in order to compute the product xy. "
                f"First, adding the two equations together eliminates y and gives 2x = {s_val} + {diff_even} = {two_x}, so x = {two_x} / 2 = {x_val}. "
                f"Next, substituting x back into the first sum equation yields y = {s_val} - {x_val} = {y_val}. "
                f"Therefore, multiplying the two integers together gives the required product xy = {x_val} * {y_val} = {ans}."
            )
            inter = (str(two_x), str(x_val), str(y_val), str(ans))
            stages = (0, 3, 2, 5)
        elif pattern == 2:
            # Linear isolation + affine evaluation: a*z + b = c -> 3z - 4
            coef = int(rng.integers(4, 11))
            n_val = int(rng.integers(6, 21))
            offset = int(rng.integers(9, 28))
            prod = coef * n_val
            rhs = prod + offset
            ans = 3 * n_val - 4
            prompt = (
                f"Solve for the integer z that satisfies {coef}z + {offset} = {rhs}, "
                f"and evaluate 3z - 4."
            )
            verbose_body = (
                f"Let's think step by step to isolate the unknown integer z and then evaluate the target expression. "
                f"First, subtracting {offset} from both sides of the equation gives {coef}z = {rhs} - {offset} = {prod}. "
                f"Next, dividing both sides of the relation by {coef} yields z = {prod} / {coef} = {n_val}. "
                f"Therefore, substituting z into the target linear expression gives 3 * {n_val} - 4 = {ans}."
            )
            inter = (str(prod), str(n_val), str(3 * n_val), str(ans))
            stages = (0, 3, 1, 5)
        elif pattern == 3:
            # Vieta's formulas: roots r, s of x^2 - Sx + P = 0 -> compute r^2 + s^2 + rs
            r_root = int(rng.integers(4, 13))
            s_root = int(rng.integers(3, 11))
            sum_rs = r_root + s_root
            prod_rs = r_root * s_root
            sq_sum = sum_rs * sum_rs
            r2_plus_s2 = sq_sum - 2 * prod_rs
            ans = r2_plus_s2 - prod_rs
            prompt = (
                f"Let r and s be the roots of the quadratic equation x^2 - {sum_rs}x + {prod_rs} = 0. "
                f"Compute the value of r^2 + s^2 - rs."
            )
            verbose_body = (
                f"Let's apply Vieta's formulas step by step rather than expanding radicals directly. "
                f"First, by Vieta's relations on x^2 - {sum_rs}x + {prod_rs} = 0, we have r + s = {sum_rs} and rs = {prod_rs}. "
                f"Next, squaring the sum identity gives (r + s)^2 = {sum_rs}^2 = {sq_sum}, so r^2 + s^2 = {sq_sum} - 2 * {prod_rs} = {r2_plus_s2}. "
                f"Therefore, subtracting one more copy of rs produces r^2 + s^2 - rs = {r2_plus_s2} - {prod_rs} = {ans}."
            )
            inter = (str(sq_sum), str(r2_plus_s2), str(prod_rs), str(ans))
            stages = (0, 3, 2, 5)
        elif pattern == 4:
            # 3-variable cyclic linear system: x+y=A, y+z=B, z+x=C -> find x*z
            x_v = int(rng.integers(4, 13))
            y_v = int(rng.integers(5, 14))
            z_v = int(rng.integers(3, 12))
            ab = x_v + y_v
            bc = y_v + z_v
            ca = z_v + x_v
            twice_sum = ab + bc + ca
            xyz_sum = twice_sum // 2
            ans = x_v * z_v
            prompt = (
                f"Three numbers x, y, and z satisfy x + y = {ab}, y + z = {bc}, and z + x = {ca}. "
                f"What is the product xz?"
            )
            verbose_body = (
                f"Let's exploit the cyclic symmetry of the three linear equations step by step. "
                f"First, summing all three equations gives 2(x + y + z) = {ab} + {bc} + {ca} = {twice_sum}, so x + y + z = {xyz_sum}. "
                f"Next, subtracting y + z = {bc} gives x = {xyz_sum} - {bc} = {x_v}, and subtracting x + y = {ab} gives z = {xyz_sum} - {ab} = {z_v}. "
                f"Therefore, multiplying x and z together yields xz = {x_v} * {z_v} = {ans}."
            )
            inter = (str(xyz_sum), str(x_v), str(z_v), str(ans))
            stages = (0, 3, 1, 5)
        else:
            # Number Theory: Euclidean division + modular congruence remainder
            mod_m = int(rng.integers(9, 17))
            quot = int(rng.integers(6, 14))
            rem = int(rng.integers(2, mod_m - 1))
            c_add = int(rng.integers(5, 18))
            mod_d = int(rng.integers(7, 13))
            n_val = mod_m * quot + rem
            lin_val = 2 * n_val + c_add
            ans = lin_val % mod_d
            prompt = (
                f"When the positive integer N is divided by {mod_m}, the quotient is {quot} and the remainder is {rem}. "
                f"What is the remainder when 2N + {c_add} is divided by {mod_d}?"
            )
            verbose_body = (
                f"Let's reconstruct the integer N from the division algorithm and then reduce modulo {mod_d} step by step. "
                f"First, by the division relation we compute N = {mod_m} * {quot} + {rem} = {n_val}. "
                f"Next, evaluating the linear expression gives 2N + {c_add} = 2 * {n_val} + {c_add} = {lin_val}. "
                f"Therefore, dividing {lin_val} by {mod_d} leaves the remainder {lin_val} mod {mod_d} = {ans}."
            )
            inter = (str(n_val), str(lin_val), str(mod_d), str(ans))
            stages = (0, 1, 4, 5)

        pruned = prune_cot_trace(verbose_body, ratio=1.0).retained_text
        examples.append(
            ReasoningExample(
                example_id=f"math_{idx:04d}",
                benchmark="MATH",
                prompt=prompt,
                verbose_cot=f"{verbose_body} #### {ans}",
                telegraphic_cot=f"{pruned} #### {ans}",
                intermediate_values=inter,
                probe_stage_labels=stages,
                final_answer=str(ans),
            )
        )

    return examples


def extract_answer_from_text(generated: str) -> str | None:
    """Extract canonical scalar answer following `####` (or trailing integer fallback)."""
    if "####" in generated:
        after = generated.split("####", 1)[1].strip()
        match = re.search(r"-?\d+", after)
        if match:
            return match.group(0)
    matches = re.findall(r"-?\d+", generated)
    return matches[-1] if matches else None


class MicroMathTokenizer:
    """Deterministic lightweight tokenizer for offline from-scratch `Qwen3-Micro` tests."""

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, texts: list[str], max_vocab: int = 256) -> None:
        base_tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "[VERBOSE]", "[TELEGRAPHIC]", "####", "\n"]
        seen = set(base_tokens)
        ordered = list(base_tokens)
        for text in texts:
            for tok in self._split(text):
                if tok not in seen and len(ordered) < max_vocab:
                    seen.add(tok)
                    ordered.append(tok)
        while len(ordered) < max_vocab:
            filler = f"<unused_{len(ordered)}>"
            ordered.append(filler)

        self.id_to_tok: list[str] = ordered
        self.tok_to_id: dict[str, int] = {t: i for i, t in enumerate(ordered)}
        self.eos_token_id: int = self.EOS_ID
        self.pad_token_id: int = self.PAD_ID

    @staticmethod
    def _split(text: str) -> list[str]:
        return re.findall(r"\[VERBOSE\]|\[TELEGRAPHIC\]|####|-?\d+|[A-Za-z_]+|[^\sA-Za-z0-9]", text)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_tok)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.tok_to_id.get(t, self.UNK_ID) for t in self._split(text)]
        if add_special_tokens:
            return [self.BOS_ID, *ids]
        return ids

    def decode(self, ids: list[int]) -> str:
        filtered = [
            self.id_to_tok[i]
            for i in ids
            if 0 <= i < len(self.id_to_tok)
            and i not in (self.PAD_ID, self.BOS_ID, self.EOS_ID)
        ]
        return " ".join(filtered)
