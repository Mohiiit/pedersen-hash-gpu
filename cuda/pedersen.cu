/**
 * CUDA Pedersen Hash Kernel
 *
 * GPU-accelerated Pedersen hash computation for Starknet's bonsai-trie.
 *
 * This kernel computes:
 *   H(a, b) = [P0 + a_low·P1 + a_high·P2 + b_low·P3 + b_high·P4]_x
 *
 * Architecture:
 * - Field elements: 4x64-bit limbs in little-endian order
 * - Points: Jacobian coordinates (X, Y, Z) for efficient computation
 * - Batch processing: one thread per hash
 */

#include <stdint.h>

//==============================================================================
// STARK Field Constants
//==============================================================================

// Prime P = 2^251 + 17·2^192 + 1
__constant__ uint64_t STARK_PRIME[4] = {
    0x0000000000000001ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL,
    0x0800000000000011ULL
};

// Montgomery constants (from Rust field/constants.rs)
__constant__ uint64_t MONT_R[4] = {
    0xffffffffffffffe1ULL,
    0xffffffffffffffffULL,
    0xffffffffffffffffULL,
    0x07fffffffffffdf0ULL
};

__constant__ uint64_t MONT_R2[4] = {
    0xfffffd737e000401ULL,
    0x00000001330fffffULL,
    0xffffffffff6f8000ULL,
    0x07ffd4ab5e008810ULL
};

// -P^(-1) mod 2^64 (Montgomery constant for reduction)
__constant__ uint64_t MONT_INV = 0xffffffffffffffffULL;

// P - 2 (for inversion exponentiation)
__constant__ uint64_t P_MINUS_2[4] = {
    0xffffffffffffffffULL,
    0xffffffffffffffffULL,
    0xffffffffffffffffULL,
    0x0800000000000010ULL
};

// Mask for extracting the low 248 bits.
__constant__ uint64_t LOW_MASK[4] = {
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0x00FFFFFFFFFFFFFFULL
};

//==============================================================================
// Pedersen Generator Points (standard representation)
//==============================================================================

__constant__ uint64_t P0_X[4] = {
    0x551fde4050ca6804ULL,
    0x716b0b1022947733ULL,
    0x00ee1b87eb599f16ULL,
    0x049ee3eba8c16007ULL
};

__constant__ uint64_t P0_Y[4] = {
    0x8b3f481e3aaa0f1aULL,
    0xc96b10228bf7b795ULL,
    0x4759ebe3da9e1df0ULL,
    0x06669b6c2df663daULL
};

__constant__ uint64_t P1_X[4] = {
    0x1080d17957ebe47bULL,
    0x8fa8120b6d56eb0cULL,
    0x969c748655fca9e5ULL,
    0x0234287dcbaffe7fULL
};

__constant__ uint64_t P1_Y[4] = {
    0x3d723d8bc943cfcaULL,
    0xdeacfd9b0d1819e0ULL,
    0x7beced415a40f0c7ULL,
    0x01ef15c18599971bULL
};

__constant__ uint64_t P2_X[4] = {
    0xb7a6932dba8aa378ULL,
    0x99099ec1de5e3018ULL,
    0x3f9dab2656558f33ULL,
    0x04fa56f376c83db3ULL
};

__constant__ uint64_t P2_Y[4] = {
    0x3aa372f0bd2d6997ULL,
    0x40c690c74709e90fULL,
    0x764910f75b45f74bULL,
    0x04ba4cc166be8decULL
};

__constant__ uint64_t P3_X[4] = {
    0x3aa372f0bd2d6997ULL,
    0x40c690c74709e90fULL,
    0x764910f75b45f74bULL,
    0x04ba4cc166be8decULL
};

__constant__ uint64_t P3_Y[4] = {
    0x6aab0cdb4f5f4a9bULL,
    0xe4c9a6ad8b2e3c6eULL,
    0x99e05d8f833cdf7aULL,
    0x06a0edc3bda0e8eaULL
};

__constant__ uint64_t P4_X[4] = {
    0xd36ff12c49a58202ULL,
    0x2ca65048d53fb325ULL,
    0x6e44cca8f61a63bbULL,
    0x054302dcb0e6cc1cULL
};

__constant__ uint64_t P4_Y[4] = {
    0x5f4e4b9fde7f7d2bULL,
    0x5c9b7b06bb92b2a0ULL,
    0xbc06547d70e98ac8ULL,
    0x064a0fb632ca0548ULL
};

//==============================================================================
// Field Element Type
//==============================================================================

typedef struct {
    uint64_t limbs[4];
} FieldElement;

//==============================================================================
// Affine Point Type
//==============================================================================

typedef struct {
    FieldElement x;
    FieldElement y;
} AffinePoint;

//==============================================================================
// Jacobian Point Type
//==============================================================================

typedef struct {
    FieldElement x;
    FieldElement y;
    FieldElement z;
} JacobianPoint;

//==============================================================================
// Field Utilities
//==============================================================================

__device__ __forceinline__ uint64_t add_with_carry(uint64_t a, uint64_t b, uint64_t* carry) {
    uint64_t sum = a + b;
    uint64_t c1 = sum < a;
    sum += *carry;
    uint64_t c2 = sum < *carry;
    *carry = c1 | c2;
    return sum;
}

__device__ __forceinline__ uint64_t sub_with_borrow(uint64_t a, uint64_t b, uint64_t* borrow) {
    uint64_t res = a - b;
    uint64_t b1 = a < b;
    uint64_t res2 = res - *borrow;
    uint64_t b2 = res < *borrow;
    *borrow = b1 | b2;
    return res2;
}

__device__ __forceinline__ void mul_add(uint64_t a, uint64_t b, uint64_t* acc, uint64_t* carry) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);

    uint64_t sum = lo + *acc;
    uint64_t c1 = sum < lo;
    sum += *carry;
    uint64_t c2 = sum < *carry;
    *acc = sum;

    *carry = hi + c1 + c2;
}

__device__ __forceinline__ void field_from_const(FieldElement* out, const uint64_t c[4]) {
    out->limbs[0] = c[0];
    out->limbs[1] = c[1];
    out->limbs[2] = c[2];
    out->limbs[3] = c[3];
}

__device__ __forceinline__ int field_is_zero(const FieldElement* a) {
    return (a->limbs[0] == 0) && (a->limbs[1] == 0) &&
           (a->limbs[2] == 0) && (a->limbs[3] == 0);
}

__device__ __forceinline__ int field_ge_prime(const FieldElement* a) {
    for (int i = 3; i >= 0; i--) {
        if (a->limbs[i] > STARK_PRIME[i]) return 1;
        if (a->limbs[i] < STARK_PRIME[i]) return 0;
    }
    return 1; // equal
}

//==============================================================================
// Field Arithmetic (Montgomery)
//==============================================================================

// Add two field elements with reduction
__device__ __forceinline__ void field_add(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint64_t carry = 0;

    for (int i = 0; i < 4; i++) {
        result->limbs[i] = add_with_carry(a->limbs[i], b->limbs[i], &carry);
    }

    if (carry || field_ge_prime(result)) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            result->limbs[i] = sub_with_borrow(result->limbs[i], STARK_PRIME[i], &borrow);
        }
    }
}

// Subtract two field elements
__device__ __forceinline__ void field_sub(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint64_t borrow = 0;

    for (int i = 0; i < 4; i++) {
        result->limbs[i] = sub_with_borrow(a->limbs[i], b->limbs[i], &borrow);
    }

    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            result->limbs[i] = add_with_carry(result->limbs[i], STARK_PRIME[i], &carry);
        }
    }
}

// Montgomery reduction: reduces 512-bit t to 256-bit modulo P
__device__ __forceinline__ void montgomery_reduce(uint64_t t[8], FieldElement* out) {
    for (int i = 0; i < 4; i++) {
        uint64_t m = (uint64_t)(0ULL - t[i]); // t[i] * MONT_INV mod 2^64, MONT_INV = -1
        uint64_t carry = 0;

        for (int j = 0; j < 4; j++) {
            mul_add(m, STARK_PRIME[j], &t[i + j], &carry);
        }

        // propagate carry into higher limbs
        int k = i + 4;
        while (carry && k < 8) {
            uint64_t acc = t[k] + carry;
            carry = (acc < t[k]) ? 1 : 0;
            t[k] = acc;
            k++;
        }
    }

    out->limbs[0] = t[4];
    out->limbs[1] = t[5];
    out->limbs[2] = t[6];
    out->limbs[3] = t[7];

    if (field_ge_prime(out)) {
        FieldElement prime;
        field_from_const(&prime, STARK_PRIME);
        field_sub(out, out, &prime);
    }
}

// Multiply two field elements (Montgomery multiplication)
__device__ __forceinline__ void field_mul(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint64_t t[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            mul_add(a->limbs[i], b->limbs[j], &t[i + j], &carry);
        }

        // add carry to next limb(s)
        int k = i + 4;
        while (carry && k < 8) {
            uint64_t acc = t[k] + carry;
            carry = (acc < t[k]) ? 1 : 0;
            t[k] = acc;
            k++;
        }
    }

    montgomery_reduce(t, result);
}

// Square a field element
__device__ __forceinline__ void field_square(FieldElement* result, const FieldElement* a) {
    field_mul(result, a, a);
}

// Double a field element
__device__ __forceinline__ void field_double(FieldElement* result, const FieldElement* a) {
    field_add(result, a, a);
}

// Convert standard -> Montgomery
__device__ __forceinline__ void field_to_mont(FieldElement* out, const FieldElement* a) {
    FieldElement r2;
    field_from_const(&r2, MONT_R2);
    field_mul(out, a, &r2);
}

// Convert Montgomery -> standard
__device__ __forceinline__ void field_from_mont(FieldElement* out, const FieldElement* a) {
    FieldElement one;
    one.limbs[0] = 1; one.limbs[1] = 0; one.limbs[2] = 0; one.limbs[3] = 0;
    field_mul(out, a, &one);
}

// Field exponentiation (square-and-multiply)
__device__ __forceinline__ void field_pow(FieldElement* out, const FieldElement* base, const uint64_t exp[4]) {
    FieldElement result;
    field_from_const(&result, MONT_R); // Montgomery representation of 1

    for (int limb = 3; limb >= 0; limb--) {
        uint64_t v = exp[limb];
        for (int bit = 63; bit >= 0; bit--) {
            FieldElement tmp;
            field_square(&tmp, &result);
            result = tmp;
            if ((v >> bit) & 1ULL) {
                field_mul(&tmp, &result, base);
                result = tmp;
            }
        }
    }

    *out = result;
}

// Field inversion using Fermat's little theorem
__device__ __forceinline__ void field_inv(FieldElement* out, const FieldElement* a) {
    field_pow(out, a, P_MINUS_2);
}

//==============================================================================
// Point Operations
//==============================================================================

__device__ __forceinline__ int point_is_identity(const JacobianPoint* p) {
    return field_is_zero(&p->z);
}

__device__ __forceinline__ void point_identity(JacobianPoint* out) {
    out->x.limbs[0] = 1; out->x.limbs[1] = 0; out->x.limbs[2] = 0; out->x.limbs[3] = 0;
    out->y.limbs[0] = 1; out->y.limbs[1] = 0; out->y.limbs[2] = 0; out->y.limbs[3] = 0;
    out->z.limbs[0] = 0; out->z.limbs[1] = 0; out->z.limbs[2] = 0; out->z.limbs[3] = 0;
}

__device__ __forceinline__ void point_from_affine(JacobianPoint* out, const AffinePoint* p) {
    out->x = p->x;
    out->y = p->y;
    field_from_const(&out->z, MONT_R); // Z = 1 in Montgomery
}

// Double a point in Jacobian coordinates
__device__ __forceinline__ void point_double(JacobianPoint* result, const JacobianPoint* p) {
    if (point_is_identity(p) || field_is_zero(&p->y)) {
        point_identity(result);
        return;
    }

    FieldElement y_squared, y_fourth;
    field_square(&y_squared, &p->y);
    field_square(&y_fourth, &y_squared);

    // S = 4·X·Y^2
    FieldElement s, temp;
    field_mul(&s, &p->x, &y_squared);
    field_double(&s, &s);
    field_double(&s, &s);

    // Z^2 and Z^4
    FieldElement z_squared, z_fourth;
    field_square(&z_squared, &p->z);
    field_square(&z_fourth, &z_squared);

    // M = 3·X^2 + Z^4 (since a = 1)
    FieldElement x_squared, m;
    field_square(&x_squared, &p->x);
    field_add(&m, &x_squared, &x_squared);
    field_add(&m, &m, &x_squared);  // 3·X^2
    field_add(&m, &m, &z_fourth);   // + Z^4

    // X' = M^2 - 2·S
    FieldElement m_squared, two_s;
    field_square(&m_squared, &m);
    field_double(&two_s, &s);
    field_sub(&result->x, &m_squared, &two_s);

    // Y' = M·(S - X') - 8·Y^4
    FieldElement s_minus_x, eight_y_fourth;
    field_sub(&s_minus_x, &s, &result->x);
    field_mul(&temp, &m, &s_minus_x);
    field_double(&eight_y_fourth, &y_fourth);
    field_double(&eight_y_fourth, &eight_y_fourth);
    field_double(&eight_y_fourth, &eight_y_fourth);
    field_sub(&result->y, &temp, &eight_y_fourth);

    // Z' = 2·Y·Z
    field_mul(&temp, &p->y, &p->z);
    field_double(&result->z, &temp);
}

// Add a Jacobian point and an affine point (add-2007-bl)
__device__ __forceinline__ void point_add_affine(JacobianPoint* result, const JacobianPoint* p,
                                                 const AffinePoint* other) {
    if (point_is_identity(p)) {
        point_from_affine(result, other);
        return;
    }

    // Z1^2 and Z1^3
    FieldElement z1_squared, z1_cubed;
    field_square(&z1_squared, &p->z);
    field_mul(&z1_cubed, &z1_squared, &p->z);

    // U2 = X2·Z1^2
    FieldElement u2;
    field_mul(&u2, &other->x, &z1_squared);

    // S2 = Y2·Z1^3
    FieldElement s2;
    field_mul(&s2, &other->y, &z1_cubed);

    // H = U2 - X1
    FieldElement h;
    field_sub(&h, &u2, &p->x);

    if (field_is_zero(&h)) {
        FieldElement s2_minus_y1;
        field_sub(&s2_minus_y1, &s2, &p->y);
        if (field_is_zero(&s2_minus_y1)) {
            point_double(result, p);
        } else {
            point_identity(result);
        }
        return;
    }

    // R = S2 - Y1
    FieldElement r;
    field_sub(&r, &s2, &p->y);

    // H^2, H^3
    FieldElement h_squared, h_cubed;
    field_square(&h_squared, &h);
    field_mul(&h_cubed, &h_squared, &h);

    // X3 = R^2 - H^3 - 2·X1·H^2
    FieldElement x1_h_squared, x_prime;
    field_mul(&x1_h_squared, &p->x, &h_squared);
    FieldElement r_squared;
    field_square(&r_squared, &r);
    FieldElement two_x1_h_squared;
    field_double(&two_x1_h_squared, &x1_h_squared);
    FieldElement tmp;
    field_sub(&tmp, &r_squared, &h_cubed);
    field_sub(&x_prime, &tmp, &two_x1_h_squared);

    // Y3 = R·(X1·H^2 - X3) - Y1·H^3
    FieldElement x1_h_squared_minus_x3;
    field_sub(&x1_h_squared_minus_x3, &x1_h_squared, &x_prime);
    FieldElement r_times;
    field_mul(&r_times, &r, &x1_h_squared_minus_x3);
    FieldElement y1_h_cubed;
    field_mul(&y1_h_cubed, &p->y, &h_cubed);
    FieldElement y_prime;
    field_sub(&y_prime, &r_times, &y1_h_cubed);

    // Z3 = Z1·H
    FieldElement z_prime;
    field_mul(&z_prime, &p->z, &h);

    result->x = x_prime;
    result->y = y_prime;
    result->z = z_prime;
}

// Add two Jacobian points
__device__ __forceinline__ void point_add(JacobianPoint* result, const JacobianPoint* p,
                                          const JacobianPoint* other) {
    if (point_is_identity(p)) {
        *result = *other;
        return;
    }
    if (point_is_identity(other)) {
        *result = *p;
        return;
    }

    // Z1^2, Z1^3, Z2^2, Z2^3
    FieldElement z1_squared, z1_cubed, z2_squared, z2_cubed;
    field_square(&z1_squared, &p->z);
    field_mul(&z1_cubed, &z1_squared, &p->z);
    field_square(&z2_squared, &other->z);
    field_mul(&z2_cubed, &z2_squared, &other->z);

    // U1 = X1·Z2^2, U2 = X2·Z1^2
    FieldElement u1, u2;
    field_mul(&u1, &p->x, &z2_squared);
    field_mul(&u2, &other->x, &z1_squared);

    // S1 = Y1·Z2^3, S2 = Y2·Z1^3
    FieldElement s1, s2;
    field_mul(&s1, &p->y, &z2_cubed);
    field_mul(&s2, &other->y, &z1_cubed);

    // H = U2 - U1
    FieldElement h;
    field_sub(&h, &u2, &u1);

    if (field_is_zero(&h)) {
        FieldElement s2_minus_s1;
        field_sub(&s2_minus_s1, &s2, &s1);
        if (field_is_zero(&s2_minus_s1)) {
            point_double(result, p);
        } else {
            point_identity(result);
        }
        return;
    }

    // R = S2 - S1
    FieldElement r;
    field_sub(&r, &s2, &s1);

    // H^2, H^3
    FieldElement h_squared, h_cubed;
    field_square(&h_squared, &h);
    field_mul(&h_cubed, &h_squared, &h);

    // X3 = R^2 - H^3 - 2·U1·H^2
    FieldElement u1_h_squared;
    field_mul(&u1_h_squared, &u1, &h_squared);
    FieldElement r_squared;
    field_square(&r_squared, &r);
    FieldElement two_u1_h_squared;
    field_double(&two_u1_h_squared, &u1_h_squared);
    FieldElement tmp;
    field_sub(&tmp, &r_squared, &h_cubed);
    FieldElement x_prime;
    field_sub(&x_prime, &tmp, &two_u1_h_squared);

    // Y3 = R·(U1·H^2 - X3) - S1·H^3
    FieldElement u1_h_squared_minus_x3;
    field_sub(&u1_h_squared_minus_x3, &u1_h_squared, &x_prime);
    FieldElement r_times;
    field_mul(&r_times, &r, &u1_h_squared_minus_x3);
    FieldElement s1_h_cubed;
    field_mul(&s1_h_cubed, &s1, &h_cubed);
    FieldElement y_prime;
    field_sub(&y_prime, &r_times, &s1_h_cubed);

    // Z3 = Z1·Z2·H
    FieldElement z1z2;
    field_mul(&z1z2, &p->z, &other->z);
    FieldElement z_prime;
    field_mul(&z_prime, &z1z2, &h);

    result->x = x_prime;
    result->y = y_prime;
    result->z = z_prime;
}

// Scalar multiplication using double-and-add
__device__ __forceinline__ void scalar_mul_affine(JacobianPoint* out, const AffinePoint* point,
                                                  const FieldElement* scalar) {
    if (field_is_zero(scalar)) {
        point_identity(out);
        return;
    }

    JacobianPoint result;
    point_identity(&result);
    int found_one = 0;

    for (int limb = 3; limb >= 0; limb--) {
        uint64_t v = scalar->limbs[limb];
        for (int bit = 63; bit >= 0; bit--) {
            if (found_one) {
                JacobianPoint tmp;
                point_double(&tmp, &result);
                result = tmp;
            }

            if ((v >> bit) & 1ULL) {
                if (found_one) {
                    JacobianPoint tmp;
                    point_add_affine(&tmp, &result, point);
                    result = tmp;
                } else {
                    point_from_affine(&result, point);
                    found_one = 1;
                }
            }
        }
    }

    *out = result;
}

//==============================================================================
// Pedersen Helpers
//==============================================================================

__device__ __forceinline__ void decompose_felt(const FieldElement* fe,
                                               FieldElement* low,
                                               FieldElement* high) {
    low->limbs[0] = fe->limbs[0] & LOW_MASK[0];
    low->limbs[1] = fe->limbs[1] & LOW_MASK[1];
    low->limbs[2] = fe->limbs[2] & LOW_MASK[2];
    low->limbs[3] = fe->limbs[3] & LOW_MASK[3];

    uint64_t high_bits = fe->limbs[3] >> 56;
    high->limbs[0] = high_bits;
    high->limbs[1] = 0;
    high->limbs[2] = 0;
    high->limbs[3] = 0;
}

__device__ __forceinline__ void projective_to_affine_x(FieldElement* out, const JacobianPoint* p) {
    if (point_is_identity(p)) {
        out->limbs[0] = 0; out->limbs[1] = 0; out->limbs[2] = 0; out->limbs[3] = 0;
        return;
    }

    FieldElement z_inv, z_inv_sq, x_affine;
    field_inv(&z_inv, &p->z);
    field_mul(&z_inv_sq, &z_inv, &z_inv);
    field_mul(&x_affine, &p->x, &z_inv_sq);
    *out = x_affine;
}

//==============================================================================
// Main Pedersen Hash Kernel
//==============================================================================

/**
 * Batch Pedersen hash kernel.
 *
 * @param inputs_a  Array of 'a' values (N elements)
 * @param inputs_b  Array of 'b' values (N elements)
 * @param outputs   Array for hash results (N elements)
 * @param N         Number of hashes to compute
 */
extern "C" __global__ void pedersen_hash_batch(
    const FieldElement* __restrict__ inputs_a,
    const FieldElement* __restrict__ inputs_b,
    FieldElement* __restrict__ outputs,
    const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // Load inputs (standard representation)
    FieldElement a = inputs_a[tid];
    FieldElement b = inputs_b[tid];

    // Decompose into low/high parts
    FieldElement a_low, a_high, b_low, b_high;
    decompose_felt(&a, &a_low, &a_high);
    decompose_felt(&b, &b_low, &b_high);

    // Load generator points and convert to Montgomery
    AffinePoint p0, p1, p2, p3, p4;
    FieldElement tmp;

    field_from_const(&tmp, P0_X); field_to_mont(&p0.x, &tmp);
    field_from_const(&tmp, P0_Y); field_to_mont(&p0.y, &tmp);

    field_from_const(&tmp, P1_X); field_to_mont(&p1.x, &tmp);
    field_from_const(&tmp, P1_Y); field_to_mont(&p1.y, &tmp);

    field_from_const(&tmp, P2_X); field_to_mont(&p2.x, &tmp);
    field_from_const(&tmp, P2_Y); field_to_mont(&p2.y, &tmp);

    field_from_const(&tmp, P3_X); field_to_mont(&p3.x, &tmp);
    field_from_const(&tmp, P3_Y); field_to_mont(&p3.y, &tmp);

    field_from_const(&tmp, P4_X); field_to_mont(&p4.x, &tmp);
    field_from_const(&tmp, P4_Y); field_to_mont(&p4.y, &tmp);

    // Compute: P0 + a_low*P1 + a_high*P2 + b_low*P3 + b_high*P4
    JacobianPoint result;
    point_from_affine(&result, &p0);

    if (!field_is_zero(&a_low)) {
        JacobianPoint tmp_point;
        scalar_mul_affine(&tmp_point, &p1, &a_low);
        JacobianPoint sum;
        point_add(&sum, &result, &tmp_point);
        result = sum;
    }

    if (!field_is_zero(&a_high)) {
        JacobianPoint tmp_point;
        scalar_mul_affine(&tmp_point, &p2, &a_high);
        JacobianPoint sum;
        point_add(&sum, &result, &tmp_point);
        result = sum;
    }

    if (!field_is_zero(&b_low)) {
        JacobianPoint tmp_point;
        scalar_mul_affine(&tmp_point, &p3, &b_low);
        JacobianPoint sum;
        point_add(&sum, &result, &tmp_point);
        result = sum;
    }

    if (!field_is_zero(&b_high)) {
        JacobianPoint tmp_point;
        scalar_mul_affine(&tmp_point, &p4, &b_high);
        JacobianPoint sum;
        point_add(&sum, &result, &tmp_point);
        result = sum;
    }

    // Extract affine x-coordinate (Montgomery) and convert to standard
    FieldElement x_mont, x_std;
    projective_to_affine_x(&x_mont, &result);
    field_from_mont(&x_std, &x_mont);

    outputs[tid] = x_std;
}

//==============================================================================
// Utility Kernels
//==============================================================================

/**
 * Copies field elements from inputs to outputs.
 */
extern "C" __global__ void field_copy_batch(
    const FieldElement* __restrict__ inputs,
    FieldElement* __restrict__ outputs,
    const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    outputs[tid] = inputs[tid];
}

/**
 * Converts standard field elements to Montgomery form.
 */
extern "C" __global__ void field_to_mont_batch(
    const FieldElement* __restrict__ inputs,
    FieldElement* __restrict__ outputs,
    const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    FieldElement out;
    field_to_mont(&out, &inputs[tid]);
    outputs[tid] = out;
}

/**
 * Converts Montgomery field elements to standard form.
 */
extern "C" __global__ void field_from_mont_batch(
    const FieldElement* __restrict__ inputs,
    FieldElement* __restrict__ outputs,
    const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    FieldElement out;
    field_from_mont(&out, &inputs[tid]);
    outputs[tid] = out;
}

/**
 * Montgomery multiplication for inputs already in Montgomery form.
 */
extern "C" __global__ void field_mul_batch(
    const FieldElement* __restrict__ inputs_a,
    const FieldElement* __restrict__ inputs_b,
    FieldElement* __restrict__ outputs,
    const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    FieldElement out;
    field_mul(&out, &inputs_a[tid], &inputs_b[tid]);
    outputs[tid] = out;
}

/**
 * 64-bit multiply helper: outputs low/high parts.
 */
extern "C" __global__ void mul64hi_batch(
    const uint64_t* __restrict__ inputs_a,
    const uint64_t* __restrict__ inputs_b,
    uint64_t* __restrict__ outputs_lo,
    uint64_t* __restrict__ outputs_hi,
    const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    uint64_t a = inputs_a[tid];
    uint64_t b = inputs_b[tid];
    outputs_lo[tid] = a * b;
    outputs_hi[tid] = __umul64hi(a, b);
}

/**
 * Debug kernel: compute mul_add for each lane.
 */
extern "C" __global__ void mul_add_batch(
    const uint64_t* __restrict__ inputs_a,
    const uint64_t* __restrict__ inputs_b,
    const uint64_t* __restrict__ inputs_acc,
    const uint64_t* __restrict__ inputs_carry,
    uint64_t* __restrict__ outputs_sum,
    uint64_t* __restrict__ outputs_carry,
    const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    uint64_t acc = inputs_acc[tid];
    uint64_t carry = inputs_carry[tid];
    mul_add(inputs_a[tid], inputs_b[tid], &acc, &carry);
    outputs_sum[tid] = acc;
    outputs_carry[tid] = carry;
}

/**
 * Debug kernel for field multiplication: dumps intermediate t arrays.
 * Only thread 0 writes outputs.
 */
extern "C" __global__ void field_mul_debug(
    const FieldElement* __restrict__ inputs_a,
    const FieldElement* __restrict__ inputs_b,
    uint64_t* __restrict__ t_mul_out,
    uint64_t* __restrict__ t_reduce_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) return;

    const FieldElement* a = &inputs_a[0];
    const FieldElement* b = &inputs_b[0];

    uint64_t t[8] = {0};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            mul_add(a->limbs[i], b->limbs[j], &t[i + j], &carry);
        }
        int k = i + 4;
        while (carry && k < 8) {
            uint64_t acc = t[k] + carry;
            carry = (acc < t[k]) ? 1 : 0;
            t[k] = acc;
            k++;
        }
    }

    for (int i = 0; i < 8; i++) {
        t_mul_out[i] = t[i];
    }

    for (int i = 0; i < 4; i++) {
        uint64_t m = (uint64_t)(0ULL - t[i]);
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            mul_add(m, STARK_PRIME[j], &t[i + j], &carry);
        }
        int k = i + 4;
        while (carry && k < 8) {
            uint64_t acc = t[k] + carry;
            carry = (acc < t[k]) ? 1 : 0;
            t[k] = acc;
            k++;
        }
        for (int j = 0; j < 8; j++) {
            t_reduce_out[i * 8 + j] = t[j];
        }
    }
}

/**
 * Warm-up kernel to initialize CUDA context.
 */
extern "C" __global__ void warmup_kernel() {
    // Empty kernel to warm up CUDA runtime
}
