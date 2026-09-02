#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <openssl/evp.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static const unsigned char PREFIX[] = "QSA-QUATERNION-FEISTEL-V1";

static int round_function(EVP_MD_CTX *ctx,
                          const unsigned char key[32],
                          int round_index,
                          const unsigned char right[8],
                          unsigned char out[8]) {
    unsigned char round_be[2];
    round_be[0] = (unsigned char)((round_index >> 8) & 0xff);
    round_be[1] = (unsigned char)(round_index & 0xff);
    if (EVP_DigestInit_ex(ctx, EVP_shake256(), NULL) != 1) return 0;
    if (EVP_DigestUpdate(ctx, PREFIX, sizeof(PREFIX) - 1) != 1) return 0;
    if (EVP_DigestUpdate(ctx, round_be, sizeof(round_be)) != 1) return 0;
    if (EVP_DigestUpdate(ctx, key, 32) != 1) return 0;
    if (EVP_DigestUpdate(ctx, right, 8) != 1) return 0;
    if (EVP_DigestFinalXOF(ctx, out, 8) != 1) return 0;
    return 1;
}

int qfeistel_encrypt_blocks(const unsigned char *input,
                            size_t length,
                            const unsigned char key[32],
                            int rounds,
                            unsigned char *output) {
    if (!input || !key || !output || (length % 16) != 0 || rounds <= 0) return -1;
    size_t blocks = length / 16;
    int failed = 0;
    #pragma omp parallel
    {
        EVP_MD_CTX *ctx = EVP_MD_CTX_new();
        if (!ctx) {
            #pragma omp atomic write
            failed = 1;
        }
        #pragma omp for schedule(static)
        for (size_t block = 0; block < blocks; ++block) {
            if (!ctx) continue;
            unsigned char left[8], right[8], f[8], next_left[8], next_right[8];
            const unsigned char *src = input + block * 16;
            memcpy(left, src, 8);
            memcpy(right, src + 8, 8);
            for (int r = 0; r < rounds; ++r) {
                if (!round_function(ctx, key, r, right, f)) {
                    #pragma omp atomic write
                    failed = 1;
                    break;
                }
                memcpy(next_left, right, 8);
                for (int i = 0; i < 8; ++i) next_right[i] = (unsigned char)(left[i] ^ f[i]);
                memcpy(left, next_left, 8);
                memcpy(right, next_right, 8);
            }
            unsigned char *dst = output + block * 16;
            memcpy(dst, left, 8);
            memcpy(dst + 8, right, 8);
        }
        if (ctx) EVP_MD_CTX_free(ctx);
    }
    return failed ? -2 : 0;
}

int qfeistel_decrypt_blocks(const unsigned char *input,
                            size_t length,
                            const unsigned char key[32],
                            int rounds,
                            unsigned char *output) {
    if (!input || !key || !output || (length % 16) != 0 || rounds <= 0) return -1;
    size_t blocks = length / 16;
    int failed = 0;
    #pragma omp parallel
    {
        EVP_MD_CTX *ctx = EVP_MD_CTX_new();
        if (!ctx) {
            #pragma omp atomic write
            failed = 1;
        }
        #pragma omp for schedule(static)
        for (size_t block = 0; block < blocks; ++block) {
            if (!ctx) continue;
            unsigned char left[8], right[8], f[8], prev_left[8], prev_right[8];
            const unsigned char *src = input + block * 16;
            memcpy(left, src, 8);
            memcpy(right, src + 8, 8);
            for (int r = rounds - 1; r >= 0; --r) {
                memcpy(prev_right, left, 8);
                if (!round_function(ctx, key, r, prev_right, f)) {
                    #pragma omp atomic write
                    failed = 1;
                    break;
                }
                for (int i = 0; i < 8; ++i) prev_left[i] = (unsigned char)(right[i] ^ f[i]);
                memcpy(left, prev_left, 8);
                memcpy(right, prev_right, 8);
            }
            unsigned char *dst = output + block * 16;
            memcpy(dst, left, 8);
            memcpy(dst + 8, right, 8);
        }
        if (ctx) EVP_MD_CTX_free(ctx);
    }
    return failed ? -2 : 0;
}
