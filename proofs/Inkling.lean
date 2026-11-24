import Std

/-!
Integer scheduling, tiling, ring-safety, and bit-decode certificates for the
Inkling 2 Mini megakernel paths.
No assertion here verifies floating-point arithmetic, Metal compiler or
memory-model behavior, occupancy, or GPU performance.
-/
namespace Inkling

def hidden : Nat := 2304
def layers : Nat := 24
def loopStart : Nat := 6
def loopEnd : Nat := 18
def loopPasses : Nat := 3
def logicalVisits : Nat := 48
def heads : Nat := 24
def kvHeads : Nat := 8
def gqaRatio : Nat := 3
def headDim : Nat := 128
def relDim : Nat := 16
def relExtent : Nat := 1024
def window : Nat := 1024
def convTaps : Nat := 4
def experts : Nat := 128
def topK : Nat := 8
def intermediate : Nat := 1152
def vocab : Nat := 201024
def unpaddedVocab : Nat := 200058
def gateRank : Nat := 64

/-- Expansion of the 24 physical layers into 48 logical visits: 6 prelude,
3 * 12 looped core, 6 coda. Each entry is (state_slot, physical_layer, pass). -/
def visitSchedule : List (Nat × Nat × Nat) :=
  let prelude := (List.range loopStart).map (fun l => (l, l, 0))
  let coreLen := loopEnd - loopStart
  let loop := (List.range loopPasses).flatMap (fun pass =>
    (List.range coreLen).map (fun off =>
      (loopStart + pass * coreLen + off, loopStart + off, pass)))
  let codaBase := loopStart + loopPasses * coreLen
  let coda := (List.range (layers - loopEnd)).map (fun off =>
    (codaBase + off, loopEnd + off, 0))
  prelude ++ loop ++ coda

theorem inkling_visit_count : visitSchedule.length = logicalVisits := by decide

theorem inkling_physical_layer_bounds :
    ∀ entry ∈ visitSchedule, entry.2.1 < layers := by decide

theorem inkling_state_slot_injective :
    visitSchedule.map Prod.fst = List.range logicalVisits := by decide

/-- Gated pre-encoded decode (`d2_gate`) ring safety: if capacity >= horizon + 2,
a stale-generation forward writing slot `(pos + 1) % capacity` never overwrites
any slot `(pos - d) % capacity` read by a recomputation of `pos` (`d <= horizon`). -/
theorem d2_gate_ring_disjoint (pos d horizon capacity : Nat)
    (hcap : horizon + 2 ≤ capacity) (hd : d ≤ horizon) (hpos : d ≤ pos) :
    (pos + 1) % capacity ≠ (pos - d) % capacity := by
  intro heq
  have hpos_c : 0 < capacity := by omega
  let r := (pos - d) % capacity
  let q := (pos - d) / capacity
  have hdm : capacity * q + r = pos - d := Nat.div_add_mod (pos - d) capacity
  have hr_lt : r < capacity := Nat.mod_lt _ hpos_c
  have hplus : pos + 1 = r + (d + 1) + q * capacity := by
    rw [Nat.mul_comm q capacity]
    omega
  have hmod1 : (pos + 1) % capacity = (r + (d + 1)) % capacity := by
    rw [hplus, Nat.add_mul_mod_self_right]
  by_cases hwrap : r + (d + 1) < capacity
  · have hm : (r + (d + 1)) % capacity = r + (d + 1) := Nat.mod_eq_of_lt hwrap
    rw [hmod1, hm] at heq
    change r + (d + 1) = r at heq
    omega
  · have hsub_lt : r + (d + 1) - capacity < capacity := by omega
    have hex : r + (d + 1) = (r + (d + 1) - capacity) + 1 * capacity := by omega
    have hm : (r + (d + 1)) % capacity = r + (d + 1) - capacity := by
      calc (r + (d + 1)) % capacity
          = ((r + (d + 1) - capacity) + 1 * capacity) % capacity := congrArg (· % capacity) hex
        _ = (r + (d + 1) - capacity) % capacity := Nat.add_mul_mod_self_right _ _ _
        _ = r + (d + 1) - capacity := Nat.mod_eq_of_lt hsub_lt
    rw [hmod1, hm] at heq
    change r + (d + 1) - capacity = r at heq
    omega

/-- Packed gated token word `(gen << 20) | token` roundtrips without overflow. -/
def d2GenShift : Nat := 20
def d2TokenMask : Nat := 2^20 - 1
def d2MaxVocab : Nat := 2^20

theorem d2_token_pack_roundtrip (gen token : Nat)
    (hg : gen < 4096) (ht : token < d2MaxVocab) :
    let word := gen * 2^20 + token
    word < 2^32 ∧ word / 2^20 = gen ∧ word % 2^20 = token := by
  dsimp only
  unfold d2MaxVocab at ht
  have hd : (gen * 2^20 + token) / 2^20 = gen := by
    rw [Nat.add_comm]
    simp [Nat.add_mul_div_right _ _ (by decide : 0 < 2^20), Nat.div_eq_of_lt ht]
  have hr : (gen * 2^20 + token) % 2^20 = token := by
    simp [Nat.add_mod, Nat.mod_eq_of_lt ht]
  exact ⟨by omega, hd, hr⟩

/-- Branchless modular ring step back by `back` slots. -/
def ringBack (slot back capacity : Nat) : Nat :=
  if back ≤ slot then slot - back else slot + capacity - back

theorem ring_back_bounds (slot back capacity : Nat)
    (hs : slot < capacity) (hb : back < capacity) :
    ringBack slot back capacity < capacity := by
  unfold ringBack
  split <;> omega

theorem ring_back_eq_mod (pos back capacity : Nat)
    (hpos : back ≤ pos) (hb : back < capacity) :
    ringBack (pos % capacity) back capacity = (pos - back) % capacity := by
  have hpc : 0 < capacity := by omega
  have hsplit : pos - back + back = pos := by omega
  have hmod_pos : pos % capacity = ((pos - back) + back) % capacity := by rw [hsplit]
  rw [hmod_pos]
  unfold ringBack
  rw [Nat.add_mod, Nat.mod_eq_of_lt hb]
  have hpm : (pos - back) % capacity < capacity := Nat.mod_lt _ hpc
  by_cases hsum : (pos - back) % capacity + back < capacity
  · rw [Nat.mod_eq_of_lt hsum]
    split <;> omega
  · have hwrap : (pos - back) % capacity + back - capacity < capacity := by omega
    have heq : (pos - back) % capacity + back = ((pos - back) % capacity + back - capacity) + 1 * capacity := by omega
    have hm : ((pos - back) % capacity + back) % capacity = (pos - back) % capacity + back - capacity := by
      calc ((pos - back) % capacity + back) % capacity
          = (((pos - back) % capacity + back - capacity) + 1 * capacity) % capacity := congrArg (· % capacity) heq
        _ = ((pos - back) % capacity + back - capacity) % capacity := Nat.add_mul_mod_self_right _ _ _
        _ = (pos - back) % capacity + back - capacity := Nat.mod_eq_of_lt hwrap
    rw [hm]
    split <;> omega

theorem ring_back_ne_current (slot back capacity : Nat)
    (_hs : slot < capacity) (hb0 : 0 < back) (hb : back < capacity) :
    ringBack slot back capacity ≠ slot := by
  unfold ringBack
  split <;> omega

/-- NVFP4 E2M1 branchless IEEE-754 binary32 bit-pattern reconstruction. -/
def nvfp4BitDecode (code : Nat) : Nat :=
  let mag := code % 8
  let sign := (code / 8) * 2^31
  let magBits := if mag < 2 then mag * 0x3f000000 else 0x3f000000 + mag * 2^22
  magBits + sign

def nvfp4ExpectedBits : List Nat :=
  [0x00000000, 0x3f000000, 0x3f800000, 0x3fc00000,
   0x40000000, 0x40400000, 0x40800000, 0x40c00000,
   0x80000000, 0xbf000000, 0xbf800000, 0xbfc00000,
   0xc0000000, 0xc0400000, 0xc0800000, 0xc0c00000]

theorem nvfp4_bit_decode_exact :
    (List.range 16).map nvfp4BitDecode = nvfp4ExpectedBits := by decide

/-- NVFP4 16-lane sub-SIMD / 16-weight scale-block ownership.
Two 16-lane sub-SIMDs share a 32-lane SIMD; each lane owns one exact 16-weight
scale block per iteration (`256` weights per sub-SIMD iteration). -/
def nvfp4SubSimdLanes : Nat := 16
def nvfp4BlockElems : Nat := 16
def nvfp4SubSimdStride : Nat := nvfp4SubSimdLanes * nvfp4BlockElems

theorem nvfp4_subsimd_coverage (k width : Nat)
    (haligned : width % nvfp4BlockElems = 0) (hk : k < width) :
    let iter := k / nvfp4SubSimdStride
    let lane16 := (k / nvfp4BlockElems) % nvfp4SubSimdLanes
    let elem := k % nvfp4BlockElems
    lane16 < nvfp4SubSimdLanes ∧ elem < nvfp4BlockElems ∧
    iter * nvfp4SubSimdStride + lane16 * nvfp4BlockElems + elem = k ∧
    iter * nvfp4SubSimdStride + lane16 * nvfp4BlockElems + nvfp4BlockElems ≤ width := by
  simp only [nvfp4SubSimdStride, nvfp4SubSimdLanes, nvfp4BlockElems] at *
  omega

theorem nvfp4_subsimd_unique (i₁ l₁ e₁ i₂ l₂ e₂ : Nat)
    (hl₁ : l₁ < nvfp4SubSimdLanes) (hl₂ : l₂ < nvfp4SubSimdLanes)
    (he₁ : e₁ < nvfp4BlockElems) (he₂ : e₂ < nvfp4BlockElems)
    (heq : i₁ * nvfp4SubSimdStride + l₁ * nvfp4BlockElems + e₁ =
           i₂ * nvfp4SubSimdStride + l₂ * nvfp4BlockElems + e₂) :
    i₁ = i₂ ∧ l₁ = l₂ ∧ e₁ = e₂ := by
  simp only [nvfp4SubSimdStride, nvfp4SubSimdLanes, nvfp4BlockElems] at *
  omega

theorem nvfp4_subsimd_scale_block (iter lane16 elem : Nat) (he : elem < nvfp4BlockElems) :
    (iter * nvfp4SubSimdStride + lane16 * nvfp4BlockElems + elem) / nvfp4BlockElems =
    iter * nvfp4SubSimdLanes + lane16 := by
  simp only [nvfp4SubSimdStride, nvfp4SubSimdLanes, nvfp4BlockElems] at *
  omega

theorem nvfp4_k_divisibility :
    hidden % nvfp4SubSimdStride = 0 ∧ intermediate % nvfp4BlockElems = 0 := by decide

/-- Generalized BF16 multi-projection (up to 6 outputs) tile mapping AST,
mirroring Resident.QkvExpr so Lean's proof and Main's emitted C/Zig match exactly. -/
def maxMultiProj : Nat := 6

inductive MultiProjExpr where
  | tile | p1 | p2 | p3 | p4 | p5
  | literal (n : Nat)
  | sub (a b : MultiProjExpr)
  | ifLt (a b yes no : MultiProjExpr)

def MultiProjExpr.eval (tile p1 p2 p3 p4 p5 : Nat) : MultiProjExpr → Nat
  | .tile => tile
  | .p1 => p1
  | .p2 => p2
  | .p3 => p3
  | .p4 => p4
  | .p5 => p5
  | .literal n => n
  | .sub a b => a.eval tile p1 p2 p3 p4 p5 - b.eval tile p1 p2 p3 p4 p5
  | .ifLt a b yes no =>
      if a.eval tile p1 p2 p3 p4 p5 < b.eval tile p1 p2 p3 p4 p5 then
        yes.eval tile p1 p2 p3 p4 p5 else no.eval tile p1 p2 p3 p4 p5

def multiProjIndexExpr : MultiProjExpr :=
  .ifLt .tile .p1 (.literal 0)
    (.ifLt .tile .p2 (.literal 1)
      (.ifLt .tile .p3 (.literal 2)
        (.ifLt .tile .p4 (.literal 3)
          (.ifLt .tile .p5 (.literal 4) (.literal 5)))))

def multiProjStartExpr : MultiProjExpr :=
  .ifLt .tile .p1 (.literal 0)
    (.ifLt .tile .p2 .p1
      (.ifLt .tile .p3 .p2
        (.ifLt .tile .p4 .p3
          (.ifLt .tile .p5 .p4 .p5))))

def multiProjLocalExpr : MultiProjExpr := .sub .tile multiProjStartExpr

def multiProjIndex (tile p1 p2 p3 p4 p5 : Nat) : Nat :=
  multiProjIndexExpr.eval tile p1 p2 p3 p4 p5

def multiProjLocalTile (tile p1 p2 p3 p4 p5 : Nat) : Nat :=
  multiProjLocalExpr.eval tile p1 p2 p3 p4 p5

def multiProjPrefix (idx p1 p2 p3 p4 p5 : Nat) : Nat :=
  if idx = 0 then 0 else if idx = 1 then p1 else if idx = 2 then p2
  else if idx = 3 then p3 else if idx = 4 then p4 else p5

def multiProjCount (idx c0 c1 c2 c3 c4 c5 : Nat) : Nat :=
  if idx = 0 then c0 else if idx = 1 then c1 else if idx = 2 then c2
  else if idx = 3 then c3 else if idx = 4 then c4 else c5

theorem multi_proj_local_bounds (tile c0 c1 c2 c3 c4 c5 : Nat)
    (h : tile < c0 + c1 + c2 + c3 + c4 + c5) :
    let p1 := c0
    let p2 := p1 + c1
    let p3 := p2 + c2
    let p4 := p3 + c3
    let p5 := p4 + c4
    let idx := multiProjIndex tile p1 p2 p3 p4 p5
    let loc := multiProjLocalTile tile p1 p2 p3 p4 p5
    idx < maxMultiProj ∧
    loc < multiProjCount idx c0 c1 c2 c3 c4 c5 ∧
    multiProjPrefix idx p1 p2 p3 p4 p5 + loc = tile := by
  dsimp only
  simp only [multiProjIndex, multiProjIndexExpr, multiProjLocalTile, multiProjLocalExpr, multiProjStartExpr,
    MultiProjExpr.eval, multiProjCount, multiProjPrefix, maxMultiProj]
  by_cases h1 : tile < c0
  · simp [h1]
  · by_cases h2 : tile < c0 + c1
    · simp [h1, h2] <;> omega
    · by_cases h3 : tile < c0 + c1 + c2
      · simp [h1, h2, h3] <;> omega
      · by_cases h4 : tile < c0 + c1 + c2 + c3
        · simp [h1, h2, h3, h4] <;> omega
        · by_cases h5 : tile < c0 + c1 + c2 + c3 + c4
          · simp [h1, h2, h3, h4, h5] <;> omega
          · simp [h1, h2, h3, h4, h5] <;> omega

theorem multi_proj_inverse_coverage (idx loc c0 c1 c2 c3 c4 c5 : Nat)
    (hidx : idx < maxMultiProj) (hloc : loc < multiProjCount idx c0 c1 c2 c3 c4 c5) :
    let p1 := c0
    let p2 := p1 + c1
    let p3 := p2 + c2
    let p4 := p3 + c3
    let p5 := p4 + c4
    let g := multiProjPrefix idx p1 p2 p3 p4 p5 + loc
    g < c0 + c1 + c2 + c3 + c4 + c5 ∧
    multiProjIndex g p1 p2 p3 p4 p5 = idx ∧
    multiProjLocalTile g p1 p2 p3 p4 p5 = loc := by
  dsimp only
  unfold maxMultiProj at hidx
  rcases (show idx = 0 ∨ idx = 1 ∨ idx = 2 ∨ idx = 3 ∨ idx = 4 ∨ idx = 5 by omega) with
    rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [multiProjCount, multiProjPrefix, multiProjIndex, multiProjIndexExpr,
      multiProjLocalTile, multiProjLocalExpr, multiProjStartExpr, MultiProjExpr.eval] at * <;>
    (try simp at *) <;>
    repeat (first | exact ⟨by omega, by omega, by omega⟩ | split <;> try omega)

theorem multi_proj_u32_bounds (tile p1 p2 p3 p4 p5 total : Nat)
    (_hmono1 : p1 ≤ p2) (_hmono2 : p2 ≤ p3) (_hmono3 : p3 ≤ p4) (_hmono4 : p4 ≤ p5)
    (_hmono5 : p5 ≤ total) (ht : tile < total) (hb : total < 2^32) :
    multiProjStartExpr.eval tile p1 p2 p3 p4 p5 ≤ tile ∧
    multiProjLocalTile tile p1 p2 p3 p4 p5 < 2^32 := by
  simp only [multiProjLocalTile, multiProjLocalExpr, multiProjStartExpr, MultiProjExpr.eval]
  by_cases h1 : tile < p1
  · simp [h1] <;> omega
  · by_cases h2 : tile < p2
    · simp [h1, h2] <;> omega
    · by_cases h3 : tile < p3
      · simp [h1, h2, h3] <;> omega
      · by_cases h4 : tile < p4
        · simp [h1, h2, h3, h4] <;> omega
        · by_cases h5 : tile < p5
          · simp [h1, h2, h3, h4, h5] <;> omega
          · simp [h1, h2, h3, h4, h5] <;> omega

/-- Split-K small-M BF16 decode ownership: 8 SIMDs x 32 lanes over `stride = n / 4` float4 vectors. -/
def splitkSimds : Nat := 8
def splitkLanes : Nat := 32
def splitkThreads : Nat := splitkSimds * splitkLanes

theorem splitk_bf16_coverage (vecIdx stride : Nat) (h : vecIdx < stride) :
    let tid := vecIdx % splitkThreads
    let simd := tid / splitkLanes
    let lane := tid % splitkLanes
    let iter := vecIdx / splitkThreads
    simd < splitkSimds ∧ lane < splitkLanes ∧
    simd * splitkLanes + lane + iter * splitkThreads = vecIdx ∧
    simd * splitkLanes + lane + iter * splitkThreads < stride := by
  simp only [splitkThreads, splitkSimds, splitkLanes] at *
  omega

theorem splitk_bf16_unique (s₁ l₁ i₁ s₂ l₂ i₂ : Nat)
    (hs₁ : s₁ < splitkSimds) (hs₂ : s₂ < splitkSimds)
    (hl₁ : l₁ < splitkLanes) (hl₂ : l₂ < splitkLanes)
    (he : s₁ * splitkLanes + l₁ + i₁ * splitkThreads =
         s₂ * splitkLanes + l₂ + i₂ * splitkThreads) :
    s₁ = s₂ ∧ l₁ = l₂ ∧ i₁ = i₂ := by
  simp only [splitkThreads, splitkSimds, splitkLanes] at *
  omega

/-- Depth attention 2304 = 576 * 4 register-resident layout + 256-thread float4 layout. -/
def depthRegThreads : Nat := 576
def depthVecElems : Nat := 4
def depthVecThreads : Nat := 256
def depthVecStride : Nat := depthVecThreads * depthVecElems

theorem depth_reg_2304_exact : depthRegThreads * depthVecElems = hidden := by decide

theorem depth_reg_ownership (d : Nat) (hd : d < hidden) :
    d / depthVecElems < depthRegThreads ∧ d % depthVecElems < depthVecElems ∧
    (d / depthVecElems) * depthVecElems + d % depthVecElems = d := by
  simp only [hidden, depthRegThreads, depthVecElems] at *
  omega

theorem depth_vec_coverage (d n : Nat) (_hd : d < n) :
    let tid := (d / depthVecElems) % depthVecThreads
    let iter := (d / depthVecElems) / depthVecThreads
    let j := d % depthVecElems
    tid < depthVecThreads ∧ j < depthVecElems ∧
    (tid + iter * depthVecThreads) * depthVecElems + j = d := by
  simp only [depthVecThreads, depthVecElems] at *
  omega

/-- Relative attention GQA-3 ownership, proportional stripe partition, and global linear slot. -/
theorem gqa3_ownership (kvh localHead : Nat) (hk : kvh < kvHeads) (hl : localHead < gqaRatio) :
    kvh * gqaRatio + localHead < heads ∧
    (kvh * gqaRatio + localHead) / gqaRatio = kvh ∧
    (kvh * gqaRatio + localHead) % gqaRatio = localHead := by
  simp only [kvHeads, gqaRatio, heads] at *
  omega

def stripeStart (visible taps part : Nat) : Nat := visible * part / taps
def stripeEnd (visible taps part : Nat) : Nat := visible * (part + 1) / taps

theorem stripe_partition_bounds (visible taps part : Nat)
    (ht : 0 < taps) (hp : part < taps) :
    stripeStart visible taps 0 = 0 ∧
    stripeEnd visible taps (taps - 1) = visible ∧
    stripeEnd visible taps part = stripeStart visible taps (part + 1) ∧
    stripeStart visible taps part ≤ stripeEnd visible taps part ∧
    stripeEnd visible taps part ≤ visible := by
  unfold stripeStart stripeEnd
  refine ⟨by simp, ?_, rfl, ?_, ?_⟩
  · have hsub : taps - 1 + 1 = taps := by omega
    rw [hsub, Nat.mul_div_cancel visible ht]
  · exact Nat.div_le_div_right (Nat.mul_le_mul_left visible (Nat.le_succ part))
  · have hle : visible * (part + 1) ≤ visible * taps := Nat.mul_le_mul_left visible (by omega)
    rw [Nat.mul_comm visible taps] at hle
    exact Nat.div_le_of_le_mul hle

theorem stripe_partition_coverage (k visible taps : Nat)
    (ht : 0 < taps) (hk : k < visible) :
    ∃ part, part < taps ∧ stripeStart visible taps part ≤ k ∧ k < stripeEnd visible taps part := by
  have hv : 0 < visible := by omega
  let part := (k * taps + taps - 1) / visible
  have hp_lt : part < taps := by
    apply (Nat.div_lt_iff_lt_mul hv).mpr
    have hm : k * taps ≤ (visible - 1) * taps := Nat.mul_le_mul_right taps (by omega)
    have hexp : (visible - 1) * taps = visible * taps - taps := by
      rw [Nat.sub_mul, Nat.one_mul]
    have hm2 : k * taps ≤ visible * taps - taps := hexp ▸ hm
    have htp : taps ≤ visible * taps := Nat.le_mul_of_pos_left taps hv
    rw [Nat.mul_comm taps visible]
    omega
  refine ⟨part, hp_lt, ?_, ?_⟩
  · unfold stripeStart
    have hmul : part * visible ≤ k * taps + taps - 1 := by
      simpa [part, Nat.mul_comm] using Nat.div_mul_le_self (k * taps + taps - 1) visible
    have h3 : visible * part < (k + 1) * taps := by
      calc visible * part = part * visible := Nat.mul_comm _ _
        _ ≤ k * taps + taps - 1 := hmul
        _ < k * taps + taps := by omega
        _ = (k + 1) * taps := by rw [Nat.add_mul, Nat.one_mul]
    exact Nat.le_of_lt_succ ((Nat.div_lt_iff_lt_mul ht).mpr h3)
  · unfold stripeEnd
    have hmod_lt : (k * taps + taps - 1) % visible < visible := Nat.mod_lt _ hv
    have hdiv_add : visible * part + (k * taps + taps - 1) % visible = k * taps + taps - 1 := by
      simpa [part] using Nat.div_add_mod (k * taps + taps - 1) visible
    have hmul_le : (k + 1) * taps ≤ visible * (part + 1) := by
      have hre : visible * (part + 1) = visible * part + visible := Nat.mul_succ visible part
      have hkm : (k + 1) * taps = k * taps + taps := by rw [Nat.add_mul, Nat.one_mul]
      omega
    exact (Nat.le_div_iff_mul_le ht).mpr hmul_le

theorem stripe_partition_disjoint (visible taps p₁ p₂ : Nat)
    (hp : p₁ < p₂) :
    stripeEnd visible taps p₁ ≤ stripeStart visible taps p₂ := by
  unfold stripeEnd stripeStart
  exact Nat.div_le_div_right (Nat.mul_le_mul_left visible (by omega))

theorem global_linear_slot (t capacity : Nat) (h : t < capacity) :
    t % capacity = t :=
  Nat.mod_eq_of_lt h

/-- Prefill global-layer Far-Band boundary for key tile size `bk`:
every tile `k0 < farBandEnd firstDiag relExtent bk` is strictly causal (`k0 + off ≤ diag`)
and strictly beyond `relExtent` (`relExtent ≤ diag - (k0 + off)`) for all rows `diag ≥ firstDiag`. -/
def farBandEnd (firstDiag relExtent bk : Nat) : Nat :=
  if relExtent + bk ≤ firstDiag + 1 then ((firstDiag + 1 - relExtent) / bk) * bk else 0

theorem prefill_far_band_exact (firstDiag relExtent bk k0 off diag : Nat)
    (hbk : 0 < bk) (hk0 : k0 < farBandEnd firstDiag relExtent bk) (haligned : k0 % bk = 0)
    (hoff : off < bk) (hdiag : firstDiag ≤ diag) :
    k0 + off ≤ diag ∧ relExtent ≤ diag - (k0 + off) := by
  unfold farBandEnd at hk0
  split at hk0
  · have hdiv_le : ((firstDiag + 1 - relExtent) / bk) * bk ≤ firstDiag + 1 - relExtent :=
      Nat.div_mul_le_self (firstDiag + 1 - relExtent) bk
    have hq_lt : k0 / bk < (firstDiag + 1 - relExtent) / bk :=
      (Nat.div_lt_iff_lt_mul hbk).mpr hk0
    have hmul : k0 = (k0 / bk) * bk := by
      have hdm := Nat.div_add_mod k0 bk
      rw [haligned, Nat.add_zero, Nat.mul_comm] at hdm
      exact hdm.symm
    have hstep : k0 + bk ≤ ((firstDiag + 1 - relExtent) / bk) * bk := by
      have h1 : (k0 / bk + 1) * bk ≤ ((firstDiag + 1 - relExtent) / bk) * bk :=
        Nat.mul_le_mul_right bk (by omega)
      have h2 : (k0 / bk + 1) * bk = k0 + bk := by rw [Nat.add_mul, Nat.one_mul, ← hmul]
      omega
    omega
  · omega

/-- NVFP4 8-lane sub-SIMD / 16-weight scale-block ownership for expert down projections
(`k = 1152 = 9 * 128`, exact without idle lanes). -/
def nvfp4SubSimd8Lanes : Nat := 8
def nvfp4SubSimd8Stride : Nat := nvfp4SubSimd8Lanes * nvfp4BlockElems

theorem nvfp4_subsimd8_coverage (k width : Nat)
    (haligned : width % nvfp4BlockElems = 0) (hk : k < width) :
    let iter := k / nvfp4SubSimd8Stride
    let lane8 := (k / nvfp4BlockElems) % nvfp4SubSimd8Lanes
    let elem := k % nvfp4BlockElems
    lane8 < nvfp4SubSimd8Lanes ∧ elem < nvfp4BlockElems ∧
    iter * nvfp4SubSimd8Stride + lane8 * nvfp4BlockElems + elem = k ∧
    iter * nvfp4SubSimd8Stride + lane8 * nvfp4BlockElems + nvfp4BlockElems ≤ width := by
  simp only [nvfp4SubSimd8Stride, nvfp4SubSimd8Lanes, nvfp4BlockElems] at *
  omega

theorem nvfp4_subsimd8_unique (i₁ l₁ e₁ i₂ l₂ e₂ : Nat)
    (hl₁ : l₁ < nvfp4SubSimd8Lanes) (hl₂ : l₂ < nvfp4SubSimd8Lanes)
    (he₁ : e₁ < nvfp4BlockElems) (he₂ : e₂ < nvfp4BlockElems)
    (heq : i₁ * nvfp4SubSimd8Stride + l₁ * nvfp4BlockElems + e₁ =
           i₂ * nvfp4SubSimd8Stride + l₂ * nvfp4BlockElems + e₂) :
    i₁ = i₂ ∧ l₁ = l₂ ∧ e₁ = e₂ := by
  simp only [nvfp4SubSimd8Stride, nvfp4SubSimd8Lanes, nvfp4BlockElems] at *
  omega

theorem nvfp4_subsimd8_scale_block (iter lane8 elem : Nat) (he : elem < nvfp4BlockElems) :
    (iter * nvfp4SubSimd8Stride + lane8 * nvfp4BlockElems + elem) / nvfp4BlockElems =
    iter * nvfp4SubSimd8Lanes + lane8 := by
  simp only [nvfp4SubSimd8Stride, nvfp4SubSimd8Lanes, nvfp4BlockElems] at *
  omega

theorem nvfp4_down_k_divisibility :
    intermediate % nvfp4SubSimd8Stride = 0 := by decide

/-- Depth attention 576 threads = 18 SIMDs x 32 lanes, each owning 4 dims. -/
def depthRegSimds : Nat := 18
def depthSimdLanes : Nat := 32

theorem depth_reg_simds : depthRegThreads = depthRegSimds * depthSimdLanes := by decide

theorem depth_reg_unique (t₁ j₁ t₂ j₂ : Nat)
    (hj₁ : j₁ < depthVecElems) (hj₂ : j₂ < depthVecElems)
    (heq : t₁ * depthVecElems + j₁ = t₂ * depthVecElems + j₂) :
    t₁ = t₂ ∧ j₁ = j₂ := by
  simp only [depthVecElems] at *
  omega

/-- Prefill N=64 expert block geometry (`M=32, N=64, K=64`, 4 SIMDs x 16 cols)
and workgroup tail partition across `parts` stripes. -/
def expertBlockN64 : Nat := 64
def expertBlockSimds : Nat := 4
def expertBlockSimdCols : Nat := 16
def expertBlockM : Nat := 32

theorem expert_block_n64_divisibility :
    hidden % expertBlockN64 = 0 ∧ (2 * intermediate) % expertBlockN64 = 0 ∧
    expertBlockSimds * expertBlockSimdCols = expertBlockN64 := by decide

theorem expert_block_n64_coverage (col : Nat) (_n : Nat) (_hc : col < _n) :
    let group := col / expertBlockN64
    let simd := (col % expertBlockN64) / expertBlockSimdCols
    let localCol := col % expertBlockSimdCols
    simd < expertBlockSimds ∧ localCol < expertBlockSimdCols ∧
    group * expertBlockN64 + simd * expertBlockSimdCols + localCol = col := by
  simp only [expertBlockN64, expertBlockSimds, expertBlockSimdCols] at *
  omega

theorem expert_block_n64_unique (g₁ s₁ c₁ g₂ s₂ c₂ : Nat)
    (hs₁ : s₁ < expertBlockSimds) (hs₂ : s₂ < expertBlockSimds)
    (hc₁ : c₁ < expertBlockSimdCols) (hc₂ : c₂ < expertBlockSimdCols)
    (heq : g₁ * expertBlockN64 + s₁ * expertBlockSimdCols + c₁ =
           g₂ * expertBlockN64 + s₂ * expertBlockSimdCols + c₂) :
    g₁ = g₂ ∧ s₁ = s₂ ∧ c₁ = c₂ := by
  simp only [expertBlockN64, expertBlockSimds, expertBlockSimdCols] at *
  omega

theorem expert_tail_partition (t count parts : Nat) (hp : 0 < parts) (ht : t < count) :
    let full := (count / expertBlockM) * expertBlockM
    (t < full ∧
      let part := (t / expertBlockM) % parts
      let iter := (t / expertBlockM) / parts
      let off := t % expertBlockM
      part < parts ∧ off < expertBlockM ∧
      (part + iter * parts) * expertBlockM + off = t) ∨
    (full ≤ t ∧ t < count ∧ (full / expertBlockM) % parts < parts) := by
  dsimp only
  simp only [expertBlockM]
  by_cases hfull : t < (count / 32) * 32
  · left
    refine ⟨hfull, Nat.mod_lt _ hp, Nat.mod_lt _ (by decide), ?_⟩
    have hdiv : (t / 32) % parts + ((t / 32) / parts) * parts = t / 32 := by
      have hdm := Nat.div_add_mod (t / 32) parts
      rw [Nat.mul_comm] at hdm
      omega
    rw [hdiv]
    omega
  · right
    exact ⟨by omega, ht, Nat.mod_lt _ hp⟩

/-- Two-stage argmax partition across `argmaxGroups = 64` threadgroups. -/
def argmaxGroups : Nat := 64
def argmaxSpan (n : Nat) : Nat := (n + argmaxGroups - 1) / argmaxGroups

theorem argmax_part_coverage (k n : Nat) (hk : k < n) :
    let span := argmaxSpan n
    let group := k / span
    let off := k % span
    0 < span ∧ group < argmaxGroups ∧ off < span ∧ group * span + off = k := by
  dsimp only
  unfold argmaxSpan argmaxGroups
  have h63 : n + 64 - 1 = n + 63 := by omega
  rw [h63]
  have hspan : 0 < (n + 63) / 64 := Nat.div_pos (by omega) (by decide)
  have hbound : n ≤ ((n + 63) / 64) * 64 := by
    have hdm := Nat.div_add_mod (n + 63) 64
    have hml : (n + 63) % 64 < 64 := Nat.mod_lt _ (by decide)
    omega
  have hmul : k < 64 * ((n + 63) / 64) := by omega
  have hglt : k / ((n + 63) / 64) < 64 := (Nat.div_lt_iff_lt_mul hspan).mpr (by simpa [Nat.mul_comm] using hmul)
  refine ⟨hspan, hglt, Nat.mod_lt _ hspan, ?_⟩
  have hdm := Nat.div_add_mod k ((n + 63) / 64)
  rw [Nat.mul_comm] at hdm
  exact hdm

theorem argmax_part_unique (g₁ o₁ g₂ o₂ span : Nat)
    (ho₁ : o₁ < span) (ho₂ : o₂ < span)
    (heq : g₁ * span + o₁ = g₂ * span + o₂) :
    g₁ = g₂ ∧ o₁ = o₂ := by
  have hs : 0 < span := by omega
  have hd : (g₁ * span + o₁) / span = g₁ := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hs, Nat.div_eq_of_lt ho₁, Nat.zero_add]
  have hd2 : (g₂ * span + o₂) / span = g₂ := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hs, Nat.div_eq_of_lt ho₂, Nat.zero_add]
  have hg : g₁ = g₂ := by rw [← hd, ← hd2, heq]
  exact ⟨hg, by rw [hg] at heq; omega⟩

/-- FP8 E4M3FN IEEE-754 binary32 bit-pattern reconstruction (`mk_expert_e4m3`). -/
def e4m3BitDecode (code : Nat) : Nat :=
  let exp := (code / 8) % 16
  let mant := code % 8
  let sign := (code / 128) * 2^31
  if exp = 15 ∧ mant = 7 then
    0x7fc00000
  else if exp = 0 then
    let magBits :=
      if mant = 0 then 0
      else if mant = 1 then 0x3b000000
      else if mant < 4 then 0x3b800000 + (mant - 2) * 2^22
      else 0x3c000000 + (mant - 4) * 2^21
    magBits + sign
  else
    sign + (exp + 120) * 2^23 + mant * 2^20

theorem e4m3_bit_decode_nan :
    e4m3BitDecode 127 = 0x7fc00000 ∧ e4m3BitDecode 255 = 0x7fc00000 := by decide

theorem e4m3_bit_decode_zero :
    e4m3BitDecode 0 = 0 ∧ e4m3BitDecode 128 = 0x80000000 ∧
    e4m3BitDecode 56 = 0x3f800000 ∧ e4m3BitDecode 184 = 0xbf800000 ∧
    e4m3BitDecode 126 = 0x43e00000 ∧ e4m3BitDecode 254 = 0xc3e00000 := by decide

theorem e4m3_bit_decode_u32 (code : Nat) (hc : code < 256) :
    e4m3BitDecode code < 2^32 := by
  unfold e4m3BitDecode
  dsimp only
  by_cases h1 : (code / 8) % 16 = 15 ∧ code % 8 = 7
  · simp [h1]
  · simp only [h1, ↓reduceIte]
    by_cases h2 : (code / 8) % 16 = 0
    · simp only [h2, ↓reduceIte]
      by_cases m0 : code % 8 = 0
      · simp only [m0, ↓reduceIte]; omega
      · simp only [m0, ↓reduceIte]
        by_cases m1 : code % 8 = 1
        · simp only [m1, ↓reduceIte]; omega
        · simp only [m1, ↓reduceIte]
          by_cases m2 : code % 8 < 4 <;> simp only [m2, ↓reduceIte] <;> omega
    · simp only [h2, ↓reduceIte]; omega

theorem fp4_pair_decompose (b : Nat) (hb : b < 256) :
    b % 16 < 16 ∧ b / 16 < 16 ∧ b % 16 + (b / 16) * 16 = b := by
  omega

theorem fp4_bit_decode_u32 (code : Nat) (hc : code < 16) :
    nvfp4BitDecode code < 2^32 := by
  unfold nvfp4BitDecode
  dsimp only
  split <;> omega

theorem fp4_pair_lut_exact (b : Nat) (hb : b < 256) :
    nvfp4BitDecode (b % 16) < 2^32 ∧ nvfp4BitDecode (b / 16) < 2^32 ∧
    nvfp4BitDecode (b % 16) ∈ nvfp4ExpectedBits ∧
    nvfp4BitDecode (b / 16) ∈ nvfp4ExpectedBits := by
  have hlo : b % 16 < 16 := by omega
  have hhi : b / 16 < 16 := by omega
  refine ⟨fp4_bit_decode_u32 (b % 16) hlo, fp4_bit_decode_u32 (b / 16) hhi, ?_, ?_⟩
  · have hmem : b % 16 ∈ List.range 16 := List.mem_range.mpr hlo
    rw [← nvfp4_bit_decode_exact]
    exact List.mem_map.mpr ⟨b % 16, hmem, rfl⟩
  · have hmem : b / 16 ∈ List.range 16 := List.mem_range.mpr hhi
    rw [← nvfp4_bit_decode_exact]
    exact List.mem_map.mpr ⟨b / 16, hmem, rfl⟩

/-- Compact MoE row-block size (`qw_moe_tile.metalh`): 16 for <=16 rows, 32 for <=32, else 64. -/
def moeTileBlock (c : Nat) : Nat :=
  if c ≤ 16 then 16 else if c ≤ 32 then 32 else 64

def moeTileCode (bm : Nat) : Nat :=
  if bm = 16 then 1 else if bm = 32 then 2 else 3

def moePackTileR (tidY bm : Nat) : Nat :=
  (moeTileCode bm) * 2^24 + (tidY % 2^24)

def moeUnpackBm (packed : Nat) : Nat :=
  let code := packed / 2^24
  if code = 1 then 16 else if code = 2 then 32 else 64

def moeUnpackTidY (packed : Nat) : Nat :=
  packed % 2^24

def moeMaxTiles (experts totalPairs : Nat) : Nat :=
  min totalPairs (min experts totalPairs + (totalPairs + 63) / 64)

theorem moe_tile_pack_roundtrip (tidY bm : Nat)
    (htid : tidY < 2^24) (hbm : bm = 16 ∨ bm = 32 ∨ bm = 64) :
    let packed := moePackTileR tidY bm
    packed < 2^32 ∧ moeUnpackBm packed = bm ∧ moeUnpackTidY packed = tidY := by
  dsimp only
  unfold moePackTileR moeUnpackBm moeUnpackTidY moeTileCode
  have hmod : tidY % 2^24 = tidY := Nat.mod_eq_of_lt htid
  rw [hmod]
  have hdiv : ∀ c : Nat, (c * 2^24 + tidY) / 2^24 = c := fun c => by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by decide : 0 < 2^24), Nat.div_eq_of_lt htid, Nat.zero_add]
  have hrm : ∀ c : Nat, (c * 2^24 + tidY) % 2^24 = tidY := fun c => by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, hmod]
  rcases hbm with rfl | rfl | rfl
  · rw [show (if (16 : Nat) = 16 then 1 else if (16 : Nat) = 32 then 2 else 3) = 1 from rfl]
    rw [hdiv 1, hrm 1]
    exact ⟨by omega, rfl, rfl⟩
  · rw [show (if (32 : Nat) = 16 then 1 else if (32 : Nat) = 32 then 2 else 3) = 2 from rfl]
    rw [hdiv 2, hrm 2]
    exact ⟨by omega, rfl, rfl⟩
  · rw [show (if (64 : Nat) = 16 then 1 else if (64 : Nat) = 32 then 2 else 3) = 3 from rfl]
    rw [hdiv 3, hrm 3]
    exact ⟨by omega, rfl, rfl⟩

private def sumMoeTiles : List Nat → Nat
  | [] => 0
  | c :: cs => (if c = 0 then 0 else (c + 63) / 64) + sumMoeTiles cs

private def activeBuckets : List Nat → Nat
  | [] => 0
  | c :: cs => (if c = 0 then 0 else 1) + activeBuckets cs

private def sumFullTiles64 : List Nat → Nat
  | [] => 0
  | c :: cs => c / 64 + sumFullTiles64 cs

private theorem sum_moe_tiles_le_sum (cs : List Nat) : sumMoeTiles cs ≤ cs.sum := by
  induction cs with
  | nil => simp [sumMoeTiles]
  | cons c cs ih =>
      simp only [sumMoeTiles, List.sum_cons]
      by_cases hc : c = 0 <;> simp [hc] <;> omega

private theorem active_buckets_le_len_and_sum (cs : List Nat) :
    activeBuckets cs ≤ cs.length ∧ activeBuckets cs ≤ cs.sum := by
  induction cs with
  | nil => simp [activeBuckets]
  | cons c cs ih =>
      simp only [activeBuckets, List.length_cons, List.sum_cons]
      by_cases hc : c = 0 <;> simp [hc] <;> omega

private theorem sum_full_tiles_mul (cs : List Nat) :
    64 * sumFullTiles64 cs ≤ cs.sum := by
  induction cs with
  | nil => simp [sumFullTiles64]
  | cons c cs ih =>
      simp only [sumFullTiles64, List.sum_cons]
      omega

private theorem sum_moe_tiles_decomp (cs : List Nat) :
    sumMoeTiles cs ≤ activeBuckets cs + sumFullTiles64 cs := by
  induction cs with
  | nil => simp [sumMoeTiles, activeBuckets, sumFullTiles64]
  | cons c cs ih =>
      simp only [sumMoeTiles, activeBuckets, sumFullTiles64]
      by_cases hc : c = 0
      · simp [hc]; omega
      · simp [hc]; omega

theorem moe_tile_count_bound (cs : List Nat) :
    sumMoeTiles cs ≤ moeMaxTiles cs.length cs.sum := by
  unfold moeMaxTiles
  have h1 := sum_moe_tiles_le_sum cs
  have ⟨h2a, h2b⟩ := active_buckets_le_len_and_sum cs
  have h3 := sum_full_tiles_mul cs
  have h4 := sum_moe_tiles_decomp cs
  have h3b : sumFullTiles64 cs ≤ (cs.sum + 63) / 64 := by omega
  omega

/-- Single-launch batched SConv + QKVR ring safety: writing `(pos + t) % capacity`
for `t < batch` never collides with reading `(pos + t' - back) % capacity` for `t' < back <= horizon`. -/
theorem sconv_batch_ring_disjoint (pos t t' back horizon batch capacity : Nat)
    (hcap : horizon + batch ≤ capacity) (ht : t < batch)
    (ht' : t' < back) (hback : back ≤ horizon) (hpos : back ≤ pos + t') :
    (pos + t) % capacity ≠ (pos + t' - back) % capacity := by
  intro heq
  have hpc : 0 < capacity := by omega
  let r := (pos + t' - back) % capacity
  let q := (pos + t' - back) / capacity
  have hdm : capacity * q + r = pos + t' - back := Nat.div_add_mod _ capacity
  have hr_lt : r < capacity := Nat.mod_lt _ hpc
  let delta := t + back - t'
  have hdelta_lo : 1 ≤ delta := by omega
  have hdelta_hi : delta < capacity := by omega
  have hplus : pos + t = r + delta + q * capacity := by
    rw [Nat.mul_comm q capacity]
    omega
  have hmod1 : (pos + t) % capacity = (r + delta) % capacity := by
    rw [hplus, Nat.add_mul_mod_self_right]
  by_cases hwrap : r + delta < capacity
  · have hm : (r + delta) % capacity = r + delta := Nat.mod_eq_of_lt hwrap
    rw [hmod1, hm] at heq
    change r + delta = r at heq
    omega
  · have hsub_lt : r + delta - capacity < capacity := by omega
    have hex : r + delta = (r + delta - capacity) + 1 * capacity := by omega
    have hm : (r + delta) % capacity = r + delta - capacity := by
      calc (r + delta) % capacity
          = ((r + delta - capacity) + 1 * capacity) % capacity := congrArg (· % capacity) hex
        _ = (r + delta - capacity) % capacity := Nat.add_mul_mod_self_right _ _ _
        _ = r + delta - capacity := Nat.mod_eq_of_lt hsub_lt
    rw [hmod1, hm] at heq
    change r + delta - capacity = r at heq
    omega

/-- Inline relative bias far-band skip: if the newest key in a stripe is at
distance >= `relExtent`, every older key in `[start, start + n)` is also outside `[0, relExtent)`. -/
theorem inline_rel_skip_far (pos start n relExtent i : Nat)
    (hstart : start + n ≤ pos + 1)
    (hn : 0 < n)
    (hfar : relExtent ≤ pos - (start + n - 1))
    (hi : i < n) :
    start + i ≤ pos ∧ relExtent ≤ pos - (start + i) := by
  omega

end Inkling
