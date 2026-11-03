// Guest test for the value (non-destructive) forms of Acc/Vec32 operations.
//
// Every case computes the same expression twice:
//   * value form  :  T t = a.op(b);           (derived value, D != S)
//   * in-place    :  U u = a.copy(); u.op_(b) (explicit copy, D == S)
// and stores both into the blob.  The host compares them bit for bit, so the
// test fails if the post-RA descriptor clone or the operand order of the
// derived-value pseudo is wrong.
//
// Inputs and result slots live in the blob (x31 base), which sim_bridge dumps
// back to the host.

#include <cstdint>

#include "tile.h"

namespace {

constexpr int ROWS = 32;
constexpr int COLS = 32;
constexpr int LANES = 32;
constexpr int ACC_BYTES = ROWS * COLS * 4;
constexpr int VEC_BYTES = LANES * 4;

constexpr std::uintptr_t A_OFF = 0x0000;
constexpr std::uintptr_t B_OFF = 0x1000;
constexpr std::uintptr_t V_OFF = 0x2000;
constexpr std::uintptr_t W_OFF = 0x2100;
constexpr std::uintptr_t OUT_OFF = 0x4000;
constexpr std::uintptr_t SLOT = 0x1000;

YAI_ALWAYS_INLINE void load_acc(Acc &a, std::byte *p) {
  a.set_dtype(AccDType::F32);
  a.set_valid_zone(ROWS, COLS);
  a.set_strides(COLS * 4, 4);
  a.set_data(p);
  a.load();
}

YAI_ALWAYS_INLINE void store_acc(Acc &a, std::byte *p) {
  a.set_strides(COLS * 4, 4);
  a.set_data(p);
  a.store();
}

YAI_ALWAYS_INLINE void load_vec(Vec32 &v, std::byte *p) {
  v.set_dtype(Vec32DType::F32);
  v.set_valid_zone(LANES);
  v.set_stride(4);
  v.set_data(p);
  v.load();
}

YAI_ALWAYS_INLINE void store_vec(Vec32 &v, std::byte *p) {
  v.set_stride(4);
  v.set_data(p);
  v.store();
}

struct Slots {
  std::byte *out;
  int next = 0;
  std::byte *take() { return out + static_cast<std::uintptr_t>(next++) * SLOT; }
};

// Each case stores the value-form result and then the in-place result.
#define CASE_ACC_BIN(NAME, VALUE_EXPR, INPLACE_STMT)                 \
  {                                                                  \
    Acc a{}, b{};                                                    \
    load_acc(a, A);                                                  \
    load_acc(b, B);                                                  \
    Acc t = (VALUE_EXPR);                                            \
    store_acc(t, slots.take());                                      \
    Acc u = a.copy();                                                \
    INPLACE_STMT;                                                    \
    store_acc(u, slots.take());                                      \
  }

#define CASE_ACC_UN(NAME, VALUE_EXPR, INPLACE_STMT)                  \
  {                                                                  \
    Acc a{};                                                         \
    load_acc(a, A);                                                  \
    Acc t = (VALUE_EXPR);                                            \
    store_acc(t, slots.take());                                      \
    Acc u = a.copy();                                                \
    INPLACE_STMT;                                                    \
    store_acc(u, slots.take());                                      \
  }

#define CASE_VEC_BIN(NAME, VALUE_EXPR, INPLACE_STMT)                 \
  {                                                                  \
    Vec32 a{}, b{};                                                  \
    load_vec(a, V);                                                  \
    load_vec(b, W);                                                  \
    Vec32 t = (VALUE_EXPR);                                          \
    store_vec(t, slots.take());                                      \
    Vec32 u = a.copy();                                              \
    INPLACE_STMT;                                                    \
    store_vec(u, slots.take());                                      \
  }

#define CASE_VEC_UN(NAME, VALUE_EXPR, INPLACE_STMT)                  \
  {                                                                  \
    Vec32 a{};                                                       \
    load_vec(a, V);                                                  \
    Vec32 t = (VALUE_EXPR);                                          \
    store_vec(t, slots.take());                                      \
    Vec32 u = a.copy();                                              \
    INPLACE_STMT;                                                    \
    store_vec(u, slots.take());                                      \
  }

} // namespace

int main() {
  std::uintptr_t base;
  asm volatile("mv %0, x31" : "=r"(base));
  std::byte *blob = reinterpret_cast<std::byte *>(base);
  std::byte *A = blob + A_OFF;
  std::byte *B = blob + B_OFF;
  std::byte *V = blob + V_OFF;
  std::byte *W = blob + W_OFF;
  Slots slots{blob + OUT_OFF};

  // 0/1: Acc + Acc
  CASE_ACC_BIN("add", a.add(b), u.add_(b));
  // 2/3: Acc - Acc
  CASE_ACC_BIN("sub", a.sub(b), u.sub_(b));
  // 4/5: Acc * Acc
  CASE_ACC_BIN("mul", a.mul(b), u.mul_(b));
  // 6/7: exp (value form is two instructions: amulx + aexp2 on the result)
  CASE_ACC_UN("exp", a.exp(), u.exp_());
  // 8/9:  Acc * vec byrow
  {
    Acc a{};
    Vec32 v{};
    load_acc(a, A);
    load_vec(v, V);
    Acc t = a.mul(v, true);
    store_acc(t, slots.take());
    Acc u = a.copy();
    u.mul_(v, true);
    store_acc(u, slots.take());
  }
  // 10/11: Acc * vec bycol
  {
    Acc a{};
    Vec32 v{};
    load_acc(a, A);
    load_vec(v, V);
    Acc t = a.mul(v, false);
    store_acc(t, slots.take());
    Acc u = a.copy();
    u.mul_(v, false);
    store_acc(u, slots.take());
  }
  // 12/13: Acc - vec byrow
  {
    Acc a{};
    Vec32 v{};
    load_acc(a, A);
    load_vec(v, V);
    Acc t = a.sub(v, true);
    store_acc(t, slots.take());
    Acc u = a.copy();
    u.sub_(v, true);
    store_acc(u, slots.take());
  }
  // 14/15: Acc * scalar
  CASE_ACC_UN("mul_scalar", a.mul(0.5f), u.mul_(0.5f));
  // 16/17: Acc + scalar
  CASE_ACC_UN("add_scalar", a.add(1.5f), u.add_(1.5f));

  // 18/19: Vec32 + Vec32
  CASE_VEC_BIN("vadd", a.add(b), u.add_(b));
  // 20/21: Vec32 - Vec32
  CASE_VEC_BIN("vsub", a.sub(b), u.sub_(b));
  // 22/23: Vec32 * Vec32
  CASE_VEC_BIN("vmul", a.mul(b), u.mul_(b));
  // 24/25: Vec32 exp
  CASE_VEC_UN("vexp", a.exp(), u.exp_());
  // 26/27: Vec32 * scalar
  CASE_VEC_UN("vmul_scalar", a.mul(0.25f), u.mul_(0.25f));

  return 0;
}
