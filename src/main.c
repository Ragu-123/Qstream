/*
 * QStream - main.c
 * CLI tool for QSF model inference.
 * Usage: qstream <model.qsf> [options]
 */
#include "qsf/qsf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <io.h>
  #include <fcntl.h>
#else
  #include <unistd.h>
  #include <time.h>
#endif

/* ── Global engine for signal handler ────────────────────────────── */
static QSFEngine* g_engine = NULL;

static void signal_handler(int sig) {
    (void)sig;
    if (g_engine) qsf_engine_interrupt(g_engine);
}

/* ── Token callback: prints each token as it's generated ─────────── */
static int print_token(uint32_t token_id, const char* text, void* userdata) {
    (void)token_id;
    int* token_count = (int*)userdata;
    if (text && text[0]) {
        fputs(text, stdout);
        fflush(stdout);
    }
    (*token_count)++;
    return 0;  /* 0 = continue */
}

/* ── Chat template ──────────────────────────────────────────────── */
/*
 * Common chat templates for instruction-tuned models.
 * Applied before tokenization to wrap user prompts properly.
 */
typedef enum {
    CHAT_NONE      = 0,  /* raw prompt, no wrapping */
    CHAT_MISTRAL   = 1,  /* [INST] {prompt} [/INST] */
    CHAT_LLAMA     = 2,  /* [INST] {prompt} [/INST] */
    CHAT_CHATML    = 3,  /* <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n */
    CHAT_GEMMA     = 4,  /* <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n */
    CHAT_PHI       = 5,  /* <|user|>\n{prompt}<|end|>\n<|assistant|>\n */
    CHAT_QWEN      = 6,  /* <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n */
    CHAT_AUTO      = 99, /* auto-detect from model arch */
} ChatTemplate;

static ChatTemplate detect_chat_template(uint32_t arch) {
    switch (arch) {
        case QSF_ARCH_MISTRAL:  return CHAT_MISTRAL;
        case QSF_ARCH_LLAMA:    return CHAT_LLAMA;
        case QSF_ARCH_GEMMA:    return CHAT_GEMMA;
        case QSF_ARCH_PHI:      return CHAT_PHI;
        case QSF_ARCH_QWEN:     return CHAT_QWEN;
        case QSF_ARCH_MIXTRAL:  return CHAT_MISTRAL;
        case QSF_ARCH_GPT_OSS:  return CHAT_CHATML;
        case QSF_ARCH_DEEPSEEK: return CHAT_CHATML;
        default:                return CHAT_NONE;
    }
}

static const char* apply_chat_template(ChatTemplate tmpl, const char* prompt,
                                         char* buf, size_t buf_size) {
    if (tmpl == CHAT_NONE || !prompt) return prompt;

    switch (tmpl) {
        case CHAT_MISTRAL:
        case CHAT_LLAMA:
            snprintf(buf, buf_size, "[INST] %s [/INST]", prompt);
            break;
        case CHAT_CHATML:
        case CHAT_QWEN:
            snprintf(buf, buf_size,
                     "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                     prompt);
            break;
        case CHAT_GEMMA:
            snprintf(buf, buf_size,
                     "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n",
                     prompt);
            break;
        case CHAT_PHI:
            snprintf(buf, buf_size,
                     "<|user|>\n%s<|end|>\n<|assistant|>\n", prompt);
            break;
        default:
            return prompt;
    }
    return buf;
}

/* ── Usage ───────────────────────────────────────────────────────── */
static void print_usage(const char* prog) {
    fprintf(stderr,
        "QStream - CPU-optimized LLM inference engine\n"
        "\n"
        "Usage: %s <model.qsf> [options]\n"
        "\n"
        "Options:\n"
        "  -p, --prompt TEXT      Input prompt (default: interactive mode)\n"
        "  -n, --max-tokens N     Max tokens to generate (default: 256)\n"
        "  -t, --temperature F    Sampling temperature (default: 0.7, 0=greedy)\n"
        "  -k, --top-k N          Top-K sampling (default: 40, 0=disabled)\n"
        "  --top-p F              Top-P nucleus sampling (default: 0.9)\n"
        "  --repeat-penalty F     Repeat penalty (default: 1.1, 1.0=off)\n"
        "  --seed N               RNG seed (default: 42)\n"
        "  --chat                 Enable chat template (auto-detect from model)\n"
        "  --chat-template NAME   Chat template: mistral, llama, chatml, gemma, phi\n"
        "  --ram-budget N         RAM budget in MB (default: auto)\n"
        "  --no-mmap              Disable memory mapping\n"
        "  -v, --verbose          Verbose output (use twice for debug)\n"
        "  -h, --help             Show this help\n"
        "\n"
        "Examples:\n"
        "  %s model.qsf -p \"Once upon a time\"\n"
        "  %s model.qsf --chat -p \"What is gravity?\"\n"
        "  %s model.qsf -p \"Hello\" -t 0 -n 100\n"
        "  %s model.qsf -p \"Explain\" -k 50 --top-p 0.95 --seed 0\n",
        prog, prog, prog, prog, prog);
}

/* ── Timer ───────────────────────────────────────────────────────── */
static double get_time_ms(void) {
#ifdef _WIN32
    static double freq = 0;
    if (freq == 0) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        freq = (double)f.QuadPart / 1000.0;
    }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / freq;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}

/* ── Main ────────────────────────────────────────────────────────── */
int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* Parse arguments */
    const char* model_path = NULL;
    const char* prompt = NULL;
    int max_tokens = 256;
    int use_chat = 0;
    ChatTemplate chat_template = CHAT_NONE;
    QSFSamplingConfig sampling;
    QSFEngineConfig   engine_cfg;

    qsf_sampling_config_default(&sampling);
    qsf_engine_config_default(&engine_cfg);
    sampling.repeat_penalty = 1.1f;  /* default repeat penalty on */

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            if (!model_path) { model_path = argv[i]; continue; }
            fprintf(stderr, "Error: unexpected argument '%s'\n", argv[i]);
            return 1;
        }

        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return 0;
        }
        else if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--prompt")) {
            if (++i >= argc) { fprintf(stderr, "Error: -p requires argument\n"); return 1; }
            prompt = argv[i];
        }
        else if (!strcmp(argv[i], "-n") || !strcmp(argv[i], "--max-tokens")) {
            if (++i >= argc) { fprintf(stderr, "Error: -n requires argument\n"); return 1; }
            max_tokens = atoi(argv[i]);
        }
        else if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "--temperature")) {
            if (++i >= argc) { fprintf(stderr, "Error: -t requires argument\n"); return 1; }
            sampling.temperature = (float)atof(argv[i]);
        }
        else if (!strcmp(argv[i], "-k") || !strcmp(argv[i], "--top-k")) {
            if (++i >= argc) { fprintf(stderr, "Error: -k requires argument\n"); return 1; }
            sampling.top_k = atoi(argv[i]);
        }
        else if (!strcmp(argv[i], "--top-p")) {
            if (++i >= argc) { fprintf(stderr, "Error: --top-p requires argument\n"); return 1; }
            sampling.top_p = (float)atof(argv[i]);
        }
        else if (!strcmp(argv[i], "--repeat-penalty")) {
            if (++i >= argc) { fprintf(stderr, "Error: --repeat-penalty requires argument\n"); return 1; }
            sampling.repeat_penalty = (float)atof(argv[i]);
        }
        else if (!strcmp(argv[i], "--seed")) {
            if (++i >= argc) { fprintf(stderr, "Error: --seed requires argument\n"); return 1; }
            sampling.seed = (uint64_t)atoll(argv[i]);
        }
        else if (!strcmp(argv[i], "--chat")) {
            use_chat = 1;
            chat_template = CHAT_AUTO;
        }
        else if (!strcmp(argv[i], "--chat-template")) {
            if (++i >= argc) { fprintf(stderr, "Error: --chat-template requires argument\n"); return 1; }
            use_chat = 1;
            if (!strcmp(argv[i], "mistral"))      chat_template = CHAT_MISTRAL;
            else if (!strcmp(argv[i], "llama"))    chat_template = CHAT_LLAMA;
            else if (!strcmp(argv[i], "chatml"))   chat_template = CHAT_CHATML;
            else if (!strcmp(argv[i], "gemma"))    chat_template = CHAT_GEMMA;
            else if (!strcmp(argv[i], "phi"))      chat_template = CHAT_PHI;
            else if (!strcmp(argv[i], "qwen"))     chat_template = CHAT_QWEN;
            else if (!strcmp(argv[i], "none"))     { use_chat = 0; chat_template = CHAT_NONE; }
            else {
                fprintf(stderr, "Unknown chat template: %s\n"
                        "Available: mistral, llama, chatml, gemma, phi, qwen, none\n",
                        argv[i]);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--ram-budget")) {
            if (++i >= argc) { fprintf(stderr, "Error: --ram-budget requires MB\n"); return 1; }
            engine_cfg.ram_budget = (size_t)atoi(argv[i]) * 1024 * 1024;
        }
        else if (!strcmp(argv[i], "--no-mmap")) {
            engine_cfg.allow_mmap = 0;
        }
        else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")) {
            engine_cfg.verbose++;
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: no model file specified\n");
        print_usage(argv[0]);
        return 1;
    }

#ifdef _WIN32
    /* Enable UTF-8 output on Windows */
    SetConsoleOutputCP(65001);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    /* Install signal handler */
    signal(SIGINT, signal_handler);
#ifdef SIGTERM
    signal(SIGTERM, signal_handler);
#endif

    /* Create engine */
    QSFEngine engine;
    g_engine = &engine;

    double t0 = get_time_ms();

    QSFError err = qsf_engine_create(&engine, model_path, &engine_cfg);
    if (err != QSF_OK) {
        fprintf(stderr, "Error loading model: %s\n", qsf_error_string(err));
        const char* detail = qsf_get_error_detail();
        if (detail && detail[0]) {
            fprintf(stderr, "  Detail: %s\n", detail);
        }
        return 1;
    }

    double t_load = get_time_ms() - t0;

    /* Auto-detect chat template from model architecture */
    if (use_chat && chat_template == CHAT_AUTO) {
        chat_template = detect_chat_template(engine.model.header.arch);
        if (engine_cfg.verbose >= 1 && chat_template != CHAT_NONE) {
            const char* names[] = {"none","mistral","llama","chatml","gemma","phi","qwen"};
            int idx = (int)chat_template;
            if (idx >= 1 && idx <= 6) {
                fprintf(stderr, "[qstream] Chat template: %s\n", names[idx]);
            }
        }
    }

    if (engine_cfg.verbose >= 1) {
        fprintf(stderr, "[qstream] Model loaded in %.1f ms\n", t_load);
    }

    char chat_buf[8192];  /* buffer for chat-templated prompt */

    if (prompt) {
        /* Apply chat template */
        const char* final_prompt = prompt;
        if (use_chat && chat_template != CHAT_NONE) {
            final_prompt = apply_chat_template(chat_template, prompt,
                                                 chat_buf, sizeof(chat_buf));
        }

        /* Single-shot mode: generate from the supplied prompt */
        fprintf(stderr, "\n");
        int token_count = 0;
        double t_gen = get_time_ms();

        err = qsf_engine_generate(&engine, final_prompt, max_tokens,
                                   &sampling, print_token, &token_count);

        double elapsed = get_time_ms() - t_gen;

        if (err != QSF_OK && err != QSF_ERR_INTERNAL) {
            fprintf(stderr, "\nError during generation: %s\n", qsf_error_string(err));
        }

        /* Stats */
        fprintf(stderr, "\n\n--- Stats ---\n");
        fprintf(stderr, "  Tokens:    %d\n", token_count);
        fprintf(stderr, "  Time:      %.1f ms\n", elapsed);
        if (token_count > 0 && elapsed > 0) {
            fprintf(stderr, "  Speed:     %.1f tokens/sec\n",
                    token_count * 1000.0 / elapsed);
            fprintf(stderr, "  Latency:   %.1f ms/token\n",
                    elapsed / token_count);
        }
    } else {
        /* Interactive mode: REPL */
        fprintf(stderr, "[qstream] Interactive mode. Type your prompt and press Enter.\n");
        if (use_chat && chat_template != CHAT_NONE) {
            fprintf(stderr, "[qstream] Chat mode enabled.\n");
        }
        fprintf(stderr, "[qstream] Type 'quit' or press Ctrl+C to exit.\n\n");

        char line_buf[4096];
        while (1) {
            fprintf(stderr, "> ");
            fflush(stderr);

            if (!fgets(line_buf, sizeof(line_buf), stdin)) break;  /* EOF */

            /* Strip trailing newline */
            size_t len = strlen(line_buf);
            while (len > 0 && (line_buf[len-1] == '\n' || line_buf[len-1] == '\r'))
                line_buf[--len] = '\0';

            if (len == 0) continue;
            if (strcmp(line_buf, "quit") == 0 || strcmp(line_buf, "exit") == 0) break;

            /* Apply chat template */
            const char* final_prompt = line_buf;
            if (use_chat && chat_template != CHAT_NONE) {
                final_prompt = apply_chat_template(chat_template, line_buf,
                                                     chat_buf, sizeof(chat_buf));
            }

            engine.interrupted = 0;
            int token_count = 0;
            double t_gen = get_time_ms();

            err = qsf_engine_generate(&engine, final_prompt, max_tokens,
                                       &sampling, print_token, &token_count);

            double elapsed = get_time_ms() - t_gen;
            fprintf(stderr, "\n[%d tokens, %.1f ms, %.1f tok/s]\n\n",
                    token_count,
                    elapsed,
                    (token_count > 0 && elapsed > 0) ? token_count * 1000.0 / elapsed : 0.0);

            if (err != QSF_OK && err != QSF_ERR_INTERNAL) {
                fprintf(stderr, "Error: %s\n", qsf_error_string(err));
            }
        }
    }

    /* Cleanup */
    qsf_engine_free(&engine);
    g_engine = NULL;

    return (err == QSF_OK) ? 0 : 1;
}
