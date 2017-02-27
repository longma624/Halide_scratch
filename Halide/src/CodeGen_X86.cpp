#include <iostream>

#include "CodeGen_X86.h"
#include "JITModule.h"
#include "IROperator.h"
#include "IRMatch.h"
#include "Debug.h"
#include "Util.h"
#include "Var.h"
#include "Param.h"
#include "LLVM_Headers.h"
#include "IRMutator.h"

namespace Halide {
namespace Internal {

using std::vector;
using std::string;

using namespace llvm;

CodeGen_X86::CodeGen_X86(Target t) : CodeGen_Posix(t) {

    #if !(WITH_X86)
    user_error << "x86 not enabled for this build of Halide.\n";
    #endif

    user_assert(llvm_X86_enabled) << "llvm build not configured with X86 target enabled.\n";

    #if !(WITH_NATIVE_CLIENT)
    user_assert(t.os != Target::NaCl) << "llvm build not configured with native client enabled.\n";
    #endif
}

Expr _i64(Expr e) {
    return cast(Int(64, e.type().lanes()), e);
}

Expr _u64(Expr e) {
    return cast(UInt(64, e.type().lanes()), e);
}
Expr _i32(Expr e) {
    return cast(Int(32, e.type().lanes()), e);
}

Expr _u32(Expr e) {
    return cast(UInt(32, e.type().lanes()), e);
}

Expr _i16(Expr e) {
    return cast(Int(16, e.type().lanes()), e);
}

Expr _u16(Expr e) {
    return cast(UInt(16, e.type().lanes()), e);
}

Expr _i8(Expr e) {
    return cast(Int(8, e.type().lanes()), e);
}

Expr _u8(Expr e) {
    return cast(UInt(8, e.type().lanes()), e);
}

Expr _f32(Expr e) {
    return cast(Float(32, e.type().lanes()), e);
}

Expr _f64(Expr e) {
    return cast(Float(64, e.type().lanes()), e);
}


namespace {

// i32(i16_a)*i32(i16_b) +/- i32(i16_c)*i32(i16_d) can be done by
// interleaving a, c, and b, d, and then using pmaddwd. We
// recognize it here, and implement it in the initial module.
bool should_use_pmaddwd(Expr a, Expr b, vector<Expr> &result) {
    Type t = a.type();
    internal_assert(b.type() == t);

    const Mul *ma = a.as<Mul>();
    const Mul *mb = b.as<Mul>();

    if (!(ma && mb && t.is_int() && t.bits() == 32 && (t.lanes() >= 4))) {
        return false;
    }

    Type narrow = t.with_bits(16);
    vector<Expr> args = {lossless_cast(narrow, ma->a),
                         lossless_cast(narrow, ma->b),
                         lossless_cast(narrow, mb->a),
                         lossless_cast(narrow, mb->b)};
    if (!args[0].defined() || !args[1].defined() ||
        !args[2].defined() || !args[3].defined()) {
        return false;
    }

    result.swap(args);
    return true;
}

}


void CodeGen_X86::visit(const Add *op) {
    vector<Expr> matches;
    if (should_use_pmaddwd(op->a, op->b, matches)) {
        codegen(Call::make(op->type, "pmaddwd", matches, Call::Extern));
    } else {
        CodeGen_Posix::visit(op);
    }
}


void CodeGen_X86::visit(const Sub *op) {
    vector<Expr> matches;
    if (should_use_pmaddwd(op->a, op->b, matches)) {
        // Negate one of the factors in the second expression
        if (is_const(matches[2])) {
            matches[2] = -matches[2];
        } else {
            matches[3] = -matches[3];
        }
        codegen(Call::make(op->type, "pmaddwd", matches, Call::Extern));
    } else {
        CodeGen_Posix::visit(op);
    }
}

void CodeGen_X86::visit(const GT *op) {
    Type t = op->a.type();
    int bits = t.lanes() * t.bits();
    if (t.lanes() == 1 || bits % 128 == 0) {
        // LLVM is fine for native vector widths or scalars
        CodeGen_Posix::visit(op);
    } else {
        // Non-native vector widths get legalized poorly by llvm. We
        // split it up ourselves.
        Value *a = codegen(op->a), *b = codegen(op->b);

        int slice_size = 128 / t.bits();
        if (target.has_feature(Target::AVX) && bits > 128) {
            slice_size = 256 / t.bits();
        }

        vector<Value *> result;
        for (int i = 0; i < op->type.lanes(); i += slice_size) {
            Value *sa = slice_vector(a, i, slice_size);
            Value *sb = slice_vector(b, i, slice_size);
            Value *slice_value;
            if (t.is_float()) {
                slice_value = builder->CreateFCmpOGT(sa, sb);
            } else if (t.is_int()) {
                slice_value = builder->CreateICmpSGT(sa, sb);
            } else {
                slice_value = builder->CreateICmpUGT(sa, sb);
            }
            result.push_back(slice_value);
        }

        value = concat_vectors(result);
        value = slice_vector(value, 0, t.lanes());
    }
}

void CodeGen_X86::visit(const EQ *op) {
    Type t = op->a.type();
    int bits = t.lanes() * t.bits();
    if (t.lanes() == 1 || bits % 128 == 0) {
        // LLVM is fine for native vector widths or scalars
        CodeGen_Posix::visit(op);
    } else {
        // Non-native vector widths get legalized poorly by llvm. We
        // split it up ourselves.
        Value *a = codegen(op->a), *b = codegen(op->b);

        int slice_size = 128 / t.bits();
        if (target.has_feature(Target::AVX) && bits > 128) {
            slice_size = 256 / t.bits();
        }

        vector<Value *> result;
        for (int i = 0; i < op->type.lanes(); i += slice_size) {
            Value *sa = slice_vector(a, i, slice_size);
            Value *sb = slice_vector(b, i, slice_size);
            Value *slice_value;
            if (t.is_float()) {
                slice_value = builder->CreateFCmpOEQ(sa, sb);
            } else {
                slice_value = builder->CreateICmpEQ(sa, sb);
            }
            result.push_back(slice_value);
        }

        value = concat_vectors(result);
        value = slice_vector(value, 0, t.lanes());
    }
}

void CodeGen_X86::visit(const LT *op) {
    codegen(op->b > op->a);
}

void CodeGen_X86::visit(const LE *op) {
    codegen(!(op->a > op->b));
}

void CodeGen_X86::visit(const GE *op) {
    codegen(!(op->b > op->a));
}

void CodeGen_X86::visit(const NE *op) {
    codegen(!(op->a == op->b));
}

void CodeGen_X86::visit(const Select *op) {

    // LLVM doesn't correctly use pblendvb for u8 vectors that aren't
    // width 16, so we peephole optimize them to intrinsics.
    struct Pattern {
        string intrin;
        Expr pattern;
    };

    static Pattern patterns[] = {
        {"pblendvb_ult_i8x16", select(wild_u8x_ < wild_u8x_, wild_i8x_, wild_i8x_)},
        {"pblendvb_ult_i8x16", select(wild_u8x_ < wild_u8x_, wild_u8x_, wild_u8x_)},
        {"pblendvb_slt_i8x16", select(wild_i8x_ < wild_i8x_, wild_i8x_, wild_i8x_)},
        {"pblendvb_slt_i8x16", select(wild_i8x_ < wild_i8x_, wild_u8x_, wild_u8x_)},
        {"pblendvb_ule_i8x16", select(wild_u8x_ <= wild_u8x_, wild_i8x_, wild_i8x_)},
        {"pblendvb_ule_i8x16", select(wild_u8x_ <= wild_u8x_, wild_u8x_, wild_u8x_)},
        {"pblendvb_sle_i8x16", select(wild_i8x_ <= wild_i8x_, wild_i8x_, wild_i8x_)},
        {"pblendvb_sle_i8x16", select(wild_i8x_ <= wild_i8x_, wild_u8x_, wild_u8x_)},
        {"pblendvb_ne_i8x16", select(wild_u8x_ != wild_u8x_, wild_i8x_, wild_i8x_)},
        {"pblendvb_ne_i8x16", select(wild_u8x_ != wild_u8x_, wild_u8x_, wild_u8x_)},
        {"pblendvb_ne_i8x16", select(wild_i8x_ != wild_i8x_, wild_i8x_, wild_i8x_)},
        {"pblendvb_ne_i8x16", select(wild_i8x_ != wild_i8x_, wild_i8x_, wild_i8x_)},
        {"pblendvb_eq_i8x16", select(wild_u8x_ == wild_u8x_, wild_i8x_, wild_i8x_)},
        {"pblendvb_eq_i8x16", select(wild_u8x_ == wild_u8x_, wild_u8x_, wild_u8x_)},
        {"pblendvb_eq_i8x16", select(wild_i8x_ == wild_i8x_, wild_i8x_, wild_i8x_)},
        {"pblendvb_eq_i8x16", select(wild_i8x_ == wild_i8x_, wild_i8x_, wild_i8x_)},
        {"pblendvb_i8x16", select(wild_u1x_, wild_i8x_, wild_i8x_)},
        {"pblendvb_i8x16", select(wild_u1x_, wild_u8x_, wild_u8x_)}
    };

    if (target.has_feature(Target::SSE41) &&
        op->condition.type().is_vector() &&
        op->type.bits() == 8 &&
        op->type.lanes() != 16) {

        vector<Expr> matches;
        for (size_t i = 0; i < sizeof(patterns)/sizeof(patterns[0]); i++) {
            if (expr_match(patterns[i].pattern, op, matches)) {
                value = call_intrin(op->type, 16, patterns[i].intrin, matches);
                return;
            }
        }
    }

    CodeGen_Posix::visit(op);

}

void CodeGen_X86::visit(const Cast *op) {

    if (!op->type.is_vector()) {
        // We only have peephole optimizations for vectors in here.
        CodeGen_Posix::visit(op);
        return;
    }

    vector<Expr> matches;

    struct Pattern {
        bool needs_sse_41;
        bool wide_op;
        Type type;
        string intrin;
        Expr pattern;
    };

    static Pattern patterns[] = {
        {false, true, Int(8, 16), "llvm.x86.sse2.padds.b",
         _i8(clamp(wild_i16x_ + wild_i16x_, -128, 127))},
        {false, true, Int(8, 16), "llvm.x86.sse2.psubs.b",
         _i8(clamp(wild_i16x_ - wild_i16x_, -128, 127))},
        {false, true, UInt(8, 16), "llvm.x86.sse2.paddus.b",
         _u8(min(wild_u16x_ + wild_u16x_, 255))},
        {false, true, UInt(8, 16), "llvm.x86.sse2.psubus.b",
         _u8(max(wild_i16x_ - wild_i16x_, 0))},
        {false, true, Int(16, 8), "llvm.x86.sse2.padds.w",
         _i16(clamp(wild_i32x_ + wild_i32x_, -32768, 32767))},
        {false, true, Int(16, 8), "llvm.x86.sse2.psubs.w",
         _i16(clamp(wild_i32x_ - wild_i32x_, -32768, 32767))},
        {false, true, UInt(16, 8), "llvm.x86.sse2.paddus.w",
         _u16(min(wild_u32x_ + wild_u32x_, 65535))},
        {false, true, UInt(16, 8), "llvm.x86.sse2.psubus.w",
         _u16(max(wild_i32x_ - wild_i32x_, 0))},
        {false, true, Int(16, 8), "llvm.x86.sse2.pmulh.w",
         _i16((wild_i32x_ * wild_i32x_) / 65536)},
        {false, true, UInt(16, 8), "llvm.x86.sse2.pmulhu.w",
         _u16((wild_u32x_ * wild_u32x_) / 65536)},
        {false, true, UInt(8, 16), "llvm.x86.sse2.pavg.b",
         _u8(((wild_u16x_ + wild_u16x_) + 1) / 2)},
        {false, true, UInt(16, 8), "llvm.x86.sse2.pavg.w",
         _u16(((wild_u32x_ + wild_u32x_) + 1) / 2)},
        {false, false, Int(16, 8), "packssdwx8",
         _i16(clamp(wild_i32x_, -32768, 32767))},
        {false, false, Int(8, 16), "packsswbx16",
         _i8(clamp(wild_i16x_, -128, 127))},
        {false, false, UInt(8, 16), "packuswbx16",
         _u8(clamp(wild_i16x_, 0, 255))},
        {true, false, UInt(16, 8), "packusdwx8",
         _u16(clamp(wild_i32x_, 0, 65535))}
    };

    for (size_t i = 0; i < sizeof(patterns)/sizeof(patterns[0]); i++) {
        const Pattern &pattern = patterns[i];

        if (!target.has_feature(Target::SSE41) && pattern.needs_sse_41) {
            continue;
        }

        if (expr_match(pattern.pattern, op, matches)) {
            bool match = true;
            if (pattern.wide_op) {
                // Try to narrow the matches to the target type.
                for (size_t i = 0; i < matches.size(); i++) {
                    matches[i] = lossless_cast(op->type, matches[i]);
                    if (!matches[i].defined()) match = false;
                }
            }
            if (match) {
                value = call_intrin(op->type, pattern.type.lanes(), pattern.intrin, matches);
                return;
            }
        }
    }


    #if LLVM_VERSION >= 38
    // Workaround for https://llvm.org/bugs/show_bug.cgi?id=24512
    // LLVM uses a numerically unstable method for vector
    // uint32->float conversion before AVX.
    if (op->value.type().element_of() == UInt(32) &&
        op->type.is_float() &&
        op->type.is_vector() &&
        !target.has_feature(Target::AVX)) {
        Type signed_type = Int(32, op->type.lanes());

        // Convert the top 31 bits to float using the signed version
        Expr top_bits = cast(signed_type, op->value / 2);
        top_bits = cast(op->type, top_bits);

        // Convert the bottom bit
        Expr bottom_bit = cast(signed_type, op->value % 2);
        bottom_bit = cast(op->type, bottom_bit);

        // Recombine as floats
        codegen(top_bits + top_bits + bottom_bit);
        return;
    }
    #endif


    CodeGen_Posix::visit(op);
}

Expr CodeGen_X86::mulhi_shr(Expr a, Expr b, int shr) {
    Type ty = a.type();
    if (ty.is_vector() && ty.bits() == 16) {
        // We can use pmulhu for this op.
        Expr p = _u16(_u32(a) * _u32(b) / 65536);
        if (shr) {
            p = p >> shr;
        }
        return p;
    }
    return CodeGen_Posix::mulhi_shr(a, b, shr);
}

void CodeGen_X86::visit(const Min *op) {
    if (!op->type.is_vector()) {
        CodeGen_Posix::visit(op);
        return;
    }

    bool use_sse_41 = target.has_feature(Target::SSE41);
    if (op->type.element_of() == UInt(8)) {
        value = call_intrin(op->type, 16, "llvm.x86.sse2.pminu.b", {op->a, op->b});
    } else if (use_sse_41 && op->type.element_of() == Int(8)) {
        value = call_intrin(op->type, 16, "llvm.x86.sse41.pminsb", {op->a, op->b});
    } else if (op->type.element_of() == Int(16)) {
        value = call_intrin(op->type, 8, "llvm.x86.sse2.pmins.w", {op->a, op->b});
    } else if (use_sse_41 && op->type.element_of() == UInt(16)) {
        value = call_intrin(op->type, 8, "llvm.x86.sse41.pminuw", {op->a, op->b});
    } else if (use_sse_41 && op->type.element_of() == Int(32)) {
        value = call_intrin(op->type, 4, "llvm.x86.sse41.pminsd", {op->a, op->b});
    } else if (use_sse_41 && op->type.element_of() == UInt(32)) {
        value = call_intrin(op->type, 4, "llvm.x86.sse41.pminud", {op->a, op->b});
    } else if (op->type.element_of() == Float(32)) {
        if (op->type.lanes() % 8 == 0 && target.has_feature(Target::AVX)) {
            // This condition should possibly be > 4, rather than a
            // multiple of 8, but shuffling in undefs seems to work
            // poorly with avx.
            value = call_intrin(op->type, 8, "min_f32x8", {op->a, op->b});
        } else {
            value = call_intrin(op->type, 4, "min_f32x4", {op->a, op->b});
        }
    } else if (op->type.element_of() == Float(64)) {
         if (op->type.lanes() % 4 == 0 && target.has_feature(Target::AVX)) {
            value = call_intrin(op->type, 4, "min_f64x4", {op->a, op->b});
        } else {
            value = call_intrin(op->type, 2, "min_f64x2", {op->a, op->b});
        }
    } else {
        CodeGen_Posix::visit(op);
    }
}

void CodeGen_X86::visit(const Max *op) {
    if (!op->type.is_vector()) {
        CodeGen_Posix::visit(op);
        return;
    }

    bool use_sse_41 = target.has_feature(Target::SSE41);
    if (op->type.element_of() == UInt(8)) {
        value = call_intrin(op->type, 16, "llvm.x86.sse2.pmaxu.b", {op->a, op->b});
    } else if (use_sse_41 && op->type.element_of() == Int(8)) {
        value = call_intrin(op->type, 16, "llvm.x86.sse41.pmaxsb", {op->a, op->b});
    } else if (op->type.element_of() == Int(16)) {
        value = call_intrin(op->type, 8, "llvm.x86.sse2.pmaxs.w", {op->a, op->b});
    } else if (use_sse_41 && op->type.element_of() == UInt(16)) {
        value = call_intrin(op->type, 8, "llvm.x86.sse41.pmaxuw", {op->a, op->b});
    } else if (use_sse_41 && op->type.element_of() == Int(32)) {
        value = call_intrin(op->type, 4, "llvm.x86.sse41.pmaxsd", {op->a, op->b});
    } else if (use_sse_41 && op->type.element_of() == UInt(32)) {
        value = call_intrin(op->type, 4, "llvm.x86.sse41.pmaxud", {op->a, op->b});
    } else if (op->type.element_of() == Float(32)) {
        if (op->type.lanes() % 8 == 0 && target.has_feature(Target::AVX)) {
            value = call_intrin(op->type, 8, "max_f32x8", {op->a, op->b});
        } else {
            value = call_intrin(op->type, 4, "max_f32x4", {op->a, op->b});
        }
    } else if (op->type.element_of() == Float(64)) {
      if (op->type.lanes() % 4 == 0 && target.has_feature(Target::AVX)) {
            value = call_intrin(op->type, 4, "max_f64x4", {op->a, op->b});
        } else {
            value = call_intrin(op->type, 2, "max_f64x2", {op->a, op->b});
        }
    } else {
        CodeGen_Posix::visit(op);
    }
}

void CodeGen_X86::visit(const Call *op) {
    if (target.has_feature(Target::AVX2) &&
        op->is_intrinsic(Call::shift_left) &&
        op->type.is_vector() &&
        op->type.is_int() &&
        op->type.bits() < 32 &&
        !is_positive_const(op->args[0])) {

        // Left shift of negative integers is broken in some cases in
        // avx2: https://llvm.org/bugs/show_bug.cgi?id=27730

        // It needs to be normalized to a 32-bit shift, because avx2
        // doesn't have a narrower version than that. We'll just do
        // that normalization ourselves. Strangely, this seems to
        // produce better asm anyway.
        Type wider = op->type.with_bits(32);
        Expr equiv = cast(op->type,
                          cast(wider, op->args[0]) <<
                          cast(wider, op->args[1]));
        codegen(equiv);
    } else {
        CodeGen_Posix::visit(op);
    }
}

string CodeGen_X86::mcpu() const {
    if (target.has_feature(Target::AVX2)) return "haswell";
    if (target.has_feature(Target::AVX)) return "corei7-avx";
    // We want SSE4.1 but not SSE4.2, hence "penryn" rather than "corei7"
    if (target.has_feature(Target::SSE41)) return "penryn";
    // Default should not include SSSE3, hence "k8" rather than "core2"
    return "k8";
}

string CodeGen_X86::mattrs() const {
    std::string features;
    std::string separator;
    #if LLVM_VERSION >= 35
    // These attrs only exist in llvm 3.5+
    if (target.has_feature(Target::FMA)) {
        features += "+fma";
        separator = ",";
    }
    if (target.has_feature(Target::FMA4)) {
        features += separator + "+fma4";
        separator = ",";
    }
    if (target.has_feature(Target::F16C)) {
        features += separator + "+f16c";
        separator = ",";
    }
    #endif
    return features;
}

bool CodeGen_X86::use_soft_float_abi() const {
    return false;
}

int CodeGen_X86::native_vector_bits() const {
    if (target.has_feature(Target::AVX)) {
        return 256;
    } else {
        return 128;
    }
}

}}
