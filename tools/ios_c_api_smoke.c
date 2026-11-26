#include "nano.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct StreamCtx {
    uint32_t ids[256];
    size_t count;
    int print_text;
} StreamCtx;

static int on_token(uint32_t token_id, const char *utf8_piece, size_t piece_len, void *user_data) {
    StreamCtx *ctx = (StreamCtx *)user_data;
    if (ctx->count < 256) ctx->ids[ctx->count] = token_id;
    ctx->count++;
    if (ctx->print_text && utf8_piece && piece_len > 0) {
        fwrite(utf8_piece, 1, piece_len, stdout);
        fflush(stdout);
    }
    return 0;
}

static void print_stats(const char *label, const NanoStats *s) {
    double prefill_tps = s->prefill_ms > 0.0 ? ((double)s->prompt_tokens * 1000.0 / s->prefill_ms) : 0.0;
    uint32_t dec_toks = s->generated_tokens > 0 ? (s->generated_tokens - 1) : 0;
    double decode_tps = (s->decode_ms > 0.0 && dec_toks > 0) ? ((double)dec_toks * 1000.0 / s->decode_ms) : 0.0;
    printf("[%s] prompt=%u gen=%u pos=%u load=%.1fms prefill=%.1fms (%.1f tok/s) decode=%.1fms (%.1f tok/s) disp=%llu\n",
        label,
        s->prompt_tokens,
        s->generated_tokens,
        s->position,
        s->load_ms,
        s->prefill_ms,
        prefill_tps,
        s->decode_ms,
        decode_tps,
        (unsigned long long)s->dispatches);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <checkpoint_dir> [prompt] [max_tokens]\n", argv[0]);
        return 2;
    }
    const char *ckpt = argv[1];
    const char *prompt = argc >= 3 ? argv[2] : "Write a one-line Python function to add two numbers.";
    uint32_t max_tokens = argc >= 4 ? (uint32_t)strtoul(argv[3], NULL, 10) : 48u;

    NanoConfig cfg = {
        .context = 2048,
        .batch = 0,
        .preset = NULL,
        .use_tokenizer = 1,
    };
    NanoSession *sess = nano_open(ckpt, &cfg);
    if (!sess) {
        fprintf(stderr, "nano_open failed: %s\n", nano_last_error());
        return 1;
    }

    printf("nano=%s model=%s device=%s\n",
        nano_version(),
        nano_model_type(sess),
        nano_device_name(sess));

    StreamCtx text_ctx = { .count = 0, .print_text = 1 };
    NanoStats stats = {0};
    printf("reply: ");
    fflush(stdout);
    if (nano_generate_text(sess, NULL, prompt, 0, max_tokens, on_token, &text_ctx, &stats) != 0) {
        fprintf(stderr, "\nnano_generate_text failed: %s\n", nano_last_error());
        nano_close(sess);
        return 1;
    }
    printf("\n");
    print_stats("text", &stats);

    if (nano_reset(sess) != 0) {
        fprintf(stderr, "nano_reset failed: %s\n", nano_last_error());
        nano_close(sess);
        return 1;
    }

    /* Deterministic fixed token prompt for cross-target parity checks (LFM3 chat turn for "Hello"). */
    static const uint32_t fixed_prompt[] = {124894u, 124899u, 6423u, 198u, 13048u, 124900u, 198u, 124899u, 78191u, 198u};
    StreamCtx tok_ctx = { .count = 0, .print_text = 0 };
    NanoStats tok_stats = {0};
    if (nano_generate_tokens(sess, fixed_prompt, sizeof(fixed_prompt) / sizeof(fixed_prompt[0]), 16u, on_token, &tok_ctx, &tok_stats) != 0) {
        fprintf(stderr, "nano_generate_tokens failed: %s\n", nano_last_error());
        nano_close(sess);
        return 1;
    }
    print_stats("fixed_tokens", &tok_stats);
    printf("fixed_ids:");
    for (size_t i = 0; i < tok_ctx.count; i++) {
        printf("%s%u", i == 0 ? " " : ",", tok_ctx.ids[i]);
    }
    printf("\n");

    nano_close(sess);
    return 0;
}
