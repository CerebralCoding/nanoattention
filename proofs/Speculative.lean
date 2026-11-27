import Std

/-!
Standalone specification of speculative greedy prefix acceptance. No model/GPU
or floating-point claims: the caller must supply exact target greedy predictions
with the ordinary decoder's argmax/tie policy. For draft inputs d0..d(k-1), the
prediction list is `first :: after_each`: first predicts d0; batch row i predicts
d(i+1). Only predictions through the first mismatch are on the committed path.

`acceptedCount` mirrors the matched-prefix branch of src/speculative.zig's
`accept`, including the output budget and an accepted EOS. The correction/bonus
is outside this count. This is a specification proof, not a verified compilation
or a claim that batch logits equal sequential logits.
-/
namespace Speculative

def acceptedCount : List Nat → List Nat → Option Nat → Nat → Nat
  | _, _, _, 0 => 0
  | [], _, _, _ => 0
  | _, [], _, _ => 0
  | d :: ds, t :: ts, eos, budget + 1 =>
      if d = t then
        if eos = some d then 1 else acceptedCount ds ts eos budget + 1
      else 0

/-- Every accepted draft token equals the target prediction at the same position,
including early termination at EOS and truncation by the remaining budget. -/
theorem accepted_prefix_eq (draft target : List Nat) (eos : Option Nat) (budget : Nat) :
    draft.take (acceptedCount draft target eos budget) =
      target.take (acceptedCount draft target eos budget) := by
  induction budget generalizing draft target with
  | zero => simp [acceptedCount]
  | succ budget ih =>
      cases draft with
      | nil => simp [acceptedCount]
      | cons d ds =>
          cases target with
          | nil => simp [acceptedCount]
          | cons t ts =>
              by_cases equal : d = t
              · subst t
                by_cases stop : eos = some d
                · simp [acceptedCount, stop]
                · simpa [acceptedCount, stop] using congrArg (List.cons d) (ih ds ts)
              · simp [acceptedCount, equal]

/-- The accepted portion alone never exceeds the remaining output budget. -/
theorem accepted_le_budget (draft target : List Nat) (eos : Option Nat) (budget : Nat) :
    acceptedCount draft target eos budget ≤ budget := by
  induction budget generalizing draft target with
  | zero => simp [acceptedCount]
  | succ budget ih =>
      cases draft with
      | nil => simp [acceptedCount]
      | cons d ds =>
          cases target with
          | nil => simp [acceptedCount]
          | cons t ts =>
              simp only [acceptedCount]
              split
              · split
                · omega
                · exact Nat.add_le_add_right (ih ds ts) 1
              · omega

/-!
Integer-only specifications for the layer-major batch dispatch and sliding-ring
journal. These definitions describe mathematical natural numbers, not finite
machine arithmetic, generated kernel code, floating-point evaluation, execution
ordering, or physical cache restoration. Integration must separately establish
that actual dispatch sizes and arithmetic implement these definitions.
-/

def laneFor (i : Nat) : Nat := i % 32

def tokenFor (count i : Nat) : Nat := (i / 32) % count

def rowTileFor (count i : Nat) : Nat := (i / 32) / count

/-- Inverse layout: lane within SIMD, then token, then row tile. -/
def threadIndex (count rowTile token lane : Nat) : Nat :=
  (rowTile * count + token) * 32 + lane

/-- Every thread index is recovered from its decoded coordinates. -/
theorem mapping_inverse (count i : Nat) (_hc : 0 < count) :
    threadIndex count (rowTileFor count i) (tokenFor count i) (laneFor i) = i := by
  unfold threadIndex rowTileFor tokenFor laneFor
  have block := Nat.div_add_mod (i / 32) count
  have thread := Nat.div_add_mod i 32
  rw [Nat.mul_comm ((i / 32) / count) count, block]
  omega

/-- Each legal coordinate triple has a thread with exactly those coordinates. -/
theorem mapping_coverage (count rowTile token lane : Nat) (hc : 0 < count)
    (ht : token < count) (hl : lane < 32) :
    let i := threadIndex count rowTile token lane
    rowTileFor count i = rowTile ∧ tokenFor count i = token ∧ laneFor i = lane := by
  have divided : threadIndex count rowTile token lane / 32 = rowTile * count + token := by
    unfold threadIndex
    omega
  have row : (rowTile * count + token) / count = rowTile := by
    rw [Nat.mul_comm rowTile count, Nat.mul_add_div hc, Nat.div_eq_of_lt ht]
    omega
  have tok : (rowTile * count + token) % count = token := by
    rw [Nat.mul_comm rowTile count, Nat.mul_add_mod, Nat.mod_eq_of_lt ht]
  dsimp only
  refine ⟨?_, ?_, ?_⟩
  · simpa only [rowTileFor, divided] using row
  · simpa only [tokenFor, divided] using tok
  · unfold laneFor threadIndex
    omega

/-- A bounded dispatch decodes only legal row, token, and SIMD-lane indices. -/
theorem mapping_bounds (count tiles i : Nat) (hc : 0 < count)
    (hi : i < tiles * count * 32) :
    rowTileFor count i < tiles ∧ tokenFor count i < count ∧ laneFor i < 32 := by
  have block : i / 32 < tiles * count := (Nat.div_lt_iff_lt_mul (by decide)).2 hi
  refine ⟨?_, Nat.mod_lt (i / 32) hc, Nat.mod_lt i (by decide)⟩
  exact (Nat.div_lt_iff_lt_mul hc).2 block

/-- Every legal triple lies inside the corresponding finite dispatch. -/
theorem covered_index_bounds (count tiles rowTile token lane : Nat) (_hc : 0 < count)
    (hr : rowTile < tiles) (ht : token < count) (hl : lane < 32) :
    threadIndex count rowTile token lane < tiles * count * 32 := by
  have next : (rowTile + 1) * count ≤ tiles * count :=
    Nat.mul_le_mul_right count (by omega)
  rw [Nat.add_mul, Nat.one_mul] at next
  have block : rowTile * count + token < tiles * count := by omega
  unfold threadIndex
  omega

/-- Journal destination for speculative input j, relative to the starting position. -/
def ringSlot (start capacity j : Nat) : Nat := (start + j) % capacity

/-- Every journal destination is a valid ring index. -/
theorem ring_slot_bounds (start capacity j : Nat) (hc : 0 < capacity) :
    ringSlot start capacity j < capacity := Nat.mod_lt (start + j) hc

/-- At most one wrap occurs after adding any offset smaller than capacity;
therefore translating a bounded set of offsets around the ring is injective. -/
theorem ring_slots_injective (start capacity count j k : Nat)
    (hc : 0 < capacity) (hcount : count ≤ capacity)
    (hj : j < count) (hk : k < count)
    (he : ringSlot start capacity j = ringSlot start capacity k) : j = k := by
  have hjc : j < capacity := by omega
  have hkc : k < capacity := by omega
  have hb := Nat.mod_lt start hc
  have wrap (x : Nat) (hx : x < capacity + capacity) :
      x % capacity = if x < capacity then x else x - capacity := by
    split
    · exact Nat.mod_eq_of_lt (by assumption)
    · rw [Nat.mod_eq_sub_mod (a := x) (b := capacity) (by omega)]
      exact Nat.mod_eq_of_lt (by omega)
  unfold ringSlot at he
  rw [Nat.add_mod start j capacity, Nat.add_mod start k capacity,
    Nat.mod_eq_of_lt hjc, Nat.mod_eq_of_lt hkc,
    wrap (start % capacity + j) (by omega),
    wrap (start % capacity + k) (by omega)] at he
  split at he <;> split at he <;> omega

/-- Different speculative inputs never journal the same ring slot when the
batch count is no larger than the ring capacity. -/
theorem ring_slots_distinct (start capacity count j k : Nat)
    (hc : 0 < capacity) (hcount : count ≤ capacity)
    (hj : j < count) (hk : k < count) (hne : j ≠ k) :
    ringSlot start capacity j ≠ ringSlot start capacity k := by
  intro he
  exact hne (ring_slots_injective start capacity count j k hc hcount hj hk he)

/-- The configured maximum batch of 32 fits a 513-slot sliding ring. -/
theorem max_batch_fits_ring (count : Nat) (hcount : count ≤ 32) : count ≤ 513 := by
  omega

/-- Keeping only an accepted prefix cannot move the committed position beyond
its speculative endpoint, which itself must have been checked against context. -/
theorem accepted_position_bounds (start accepted count context : Nat)
    (ha : accepted ≤ count) (hc : start + count ≤ context) :
    start + accepted ≤ start + count ∧ start + count ≤ context := by
  exact ⟨Nat.add_le_add_left ha start, hc⟩

/-- An accepted journal slot is distinct from every rejected journal slot.
This separates the integer destinations; it does not prove physical restoration. -/
theorem accepted_rejected_slots_distinct (start capacity count accepted j k : Nat)
    (hc : 0 < capacity) (hcount : count ≤ capacity) (ha : accepted ≤ count)
    (hj : j < accepted) (hklo : accepted ≤ k) (hkhi : k < count) :
    ringSlot start capacity j ≠ ringSlot start capacity k := by
  apply ring_slots_distinct start capacity count j k hc hcount (by omega) hkhi
  omega

def maxBy {α : Type} (s : α → Int) : α → List α → α
  | best, [] => best
  | best, x :: xs => if s x > s best then maxBy s x xs else maxBy s best xs

/-- Strictly order-preserving transform preserves exact argmax selection. -/
theorem strict_mono_argmax_eq {α : Type} (s : α → Int) (f : Int → Int)
    (hf : ∀ a b : Int, a < b → f a < f b) (x0 : α) (xs : List α) :
    maxBy (fun a => f (s a)) x0 xs = maxBy s x0 xs := by
  induction xs generalizing x0 with
  | nil => rfl
  | cons x xs ih =>
      have hiff : f (s x) > f (s x0) ↔ s x > s x0 := by
        constructor
        · intro hlt
          by_cases h : s x > s x0
          · exact h
          · exfalso
            have hle : s x ≤ s x0 := Int.not_lt.mp h
            rcases Int.lt_or_eq_of_le hle with hlt2 | heq
            · have := hf (s x) (s x0) hlt2; omega
            · rw [heq] at hlt; omega
        · intro hlt; exact hf (s x0) (s x) hlt
      simp only [maxBy]
      by_cases h : s x > s x0
      · have hfgt : f (s x) > f (s x0) := hiff.mpr h
        simp [h, hfgt, ih x]
      · have hfn : ¬(f (s x) > f (s x0)) := fun hp => h (hiff.mp hp)
        simp [h, hfn, ih x0]

def dcutSurv (S : Nat) (c : Nat → Nat) : Nat → Nat
  | 0 => S
  | k + 1 => (dcutSurv S c k * c k) / S

def dcutUtil (S : Nat) (c : Nat → Nat) : Nat → Nat
  | 0 => 0
  | r + 1 => dcutUtil S c r + dcutSurv S c r

/-- D-cut prefix survival factors are bounded in `[0, S]` and antitone in depth `k`. -/
theorem dcut_prefix_antitone (S : Nat) (hS : 0 < S) (c : Nat → Nat) (hc : ∀ t, c t ≤ S) :
    ∀ k, dcutSurv S c (k + 1) ≤ dcutSurv S c k ∧ dcutSurv S c k ≤ S := by
  intro k
  induction k with
  | zero =>
      refine ⟨?_, Nat.le_refl S⟩
      calc dcutSurv S c 1
          = (S * c 0) / S := rfl
        _ ≤ (S * S) / S := Nat.div_le_div_right (Nat.mul_le_mul_left S (hc 0))
        _ = S := Nat.mul_div_cancel S hS
  | succ k ih =>
      have hstep : dcutSurv S c (k + 2) ≤ dcutSurv S c (k + 1) := by
        calc dcutSurv S c (k + 2)
            = (dcutSurv S c (k + 1) * c (k + 1)) / S := rfl
          _ ≤ (dcutSurv S c (k + 1) * S) / S := Nat.div_le_div_right (Nat.mul_le_mul_left _ (hc (k + 1)))
          _ = dcutSurv S c (k + 1) := Nat.mul_div_cancel _ hS
      exact ⟨hstep, Nat.le_trans ih.1 ih.2⟩

/-- Expected token utility `U(r)` under D-cut is non-decreasing and has diminishing marginal gains. -/
theorem dcut_utility_monotone_concave (S : Nat) (hS : 0 < S) (c : Nat → Nat) (hc : ∀ t, c t ≤ S) (r : Nat) :
    dcutUtil S c r ≤ dcutUtil S c (r + 1) ∧
    dcutUtil S c (r + 2) - dcutUtil S c (r + 1) ≤ dcutUtil S c (r + 1) - dcutUtil S c r := by
  have hant := dcut_prefix_antitone S hS c hc r
  constructor
  · simp only [dcutUtil]; omega
  · simp only [dcutUtil]
    have h1 : dcutUtil S c r + dcutSurv S c r + dcutSurv S c (r + 1) - (dcutUtil S c r + dcutSurv S c r) = dcutSurv S c (r + 1) := by omega
    have h2 : dcutUtil S c r + dcutSurv S c r - dcutUtil S c r = dcutSurv S c r := by omega
    rw [h1, h2]
    exact hant.1

/-- Truncating verification from block size `B` to `r` (`1 ≤ r`) preserves an exact greedy prefix. -/
theorem dcut_truncation_exact_parity (draft target : List Nat) (eos : Option Nat) (budget r : Nat) :
    (draft.take (r - 1)).take (acceptedCount (draft.take (r - 1)) target eos budget) =
      target.take (acceptedCount (draft.take (r - 1)) target eos budget) :=
  accepted_prefix_eq (draft.take (r - 1)) target eos budget

/-- Every D-cut verified batch advances strictly (`≥ 1` token emitted) and accepts at most `r - 1` draft tokens. -/
theorem dcut_verify_progress (draft target : List Nat) (eos : Option Nat) (budget r : Nat) (_hr : 1 ≤ r) :
    1 ≤ 1 + acceptedCount (draft.take (r - 1)) target eos budget ∧
    acceptedCount (draft.take (r - 1)) target eos budget ≤ r - 1 := by
  refine ⟨by omega, ?_⟩
  have hlen : ∀ (d t : List Nat) (b : Nat), acceptedCount d t eos b ≤ d.length := by
    intro d t b
    induction b generalizing d t with
    | zero => simp [acceptedCount]
    | succ b ih =>
        cases d with
        | nil => simp [acceptedCount]
        | cons d0 ds =>
            cases t with
            | nil => simp [acceptedCount]
            | cons t0 ts =>
                simp only [acceptedCount]
                split
                · split
                  · simp
                  · have hi := ih ds ts
                    simp only [List.length_cons]
                    omega
                · simp
  exact Nat.le_trans (hlen (draft.take (r - 1)) target budget) (List.length_take_le _ _)

end Speculative

#print axioms Speculative.accepted_prefix_eq
#print axioms Speculative.accepted_le_budget
#print axioms Speculative.mapping_inverse
#print axioms Speculative.mapping_coverage
#print axioms Speculative.mapping_bounds
#print axioms Speculative.covered_index_bounds
#print axioms Speculative.ring_slot_bounds
#print axioms Speculative.ring_slots_injective
#print axioms Speculative.ring_slots_distinct
#print axioms Speculative.max_batch_fits_ring
#print axioms Speculative.accepted_position_bounds
#print axioms Speculative.accepted_rejected_slots_distinct
#print axioms Speculative.strict_mono_argmax_eq
#print axioms Speculative.dcut_prefix_antitone
#print axioms Speculative.dcut_utility_monotone_concave
#print axioms Speculative.dcut_truncation_exact_parity
#print axioms Speculative.dcut_verify_progress
