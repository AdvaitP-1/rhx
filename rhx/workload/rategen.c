/*
 * rategen.c -- synthetic workload with KNOWN per-page access-rate ground truth.
 *
 * Purpose
 * -------
 * Gate 0 of the reclaim-hysteresis experiment program requires a workload whose
 * true access-rate density f(lambda) is known exactly, so that an estimator can
 * be validated against ground truth before it is trusted on real applications.
 *
 * Design
 * ------
 *   - Maps a region of N pages (file-backed MAP_SHARED by default, so pages are
 *     reclaimable as page cache WITHOUT requiring swap; anonymous is optional).
 *   - Assigns each page p an access rate lambda_p drawn from a chosen
 *     distribution with a recorded seed. The full assignment is written to disk
 *     as ground truth BEFORE any access occurs.
 *   - Drives accesses as independent Poisson processes: page p's inter-access
 *     times are Exp(lambda_p). Implemented exactly with a binary min-heap over
 *     next-access times, so the realized process is a true superposition of
 *     independent Poisson processes (not a discretized approximation).
 *   - Exposes a control FIFO so the harness can request residency snapshots via
 *     mincore(2), which gives EXACT per-page residency rather than inferring it
 *     from refault counters.
 *
 * Correctness notes
 * -----------------
 *   - Touches use a volatile read so the compiler cannot elide them.
 *   - CLOCK_MONOTONIC is used throughout; never CLOCK_REALTIME.
 *   - The PRNG is xoshiro256++ (public domain), seeded via splitmix64, so runs
 *     are exactly reproducible from the recorded seed.
 *   - Exponential sampling uses -log(u)/lambda with u in (0,1]; u == 0 is
 *     rejected so log() is always finite.
 *   - Access counts are maintained per page for the "counts" estimator; these
 *     are the sufficient statistic for a Poisson rate MLE over a known window.
 *   - If the process falls behind schedule (aggregate rate too high for the
 *     machine), it reports lag rather than silently dropping events.
 *
 * Build:  make
 * Usage:  see --help
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

/* ------------------------------------------------------------------ */
/* PRNG: splitmix64 seeding + xoshiro256++ (public domain reference)    */
/* ------------------------------------------------------------------ */

static uint64_t sm64_state;

static uint64_t splitmix64(void) {
    uint64_t z = (sm64_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint64_t xs_state[4];

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro_next(void) {
    const uint64_t result = rotl(xs_state[0] + xs_state[3], 23) + xs_state[0];
    const uint64_t t = xs_state[1] << 17;
    xs_state[2] ^= xs_state[0];
    xs_state[3] ^= xs_state[1];
    xs_state[1] ^= xs_state[2];
    xs_state[0] ^= xs_state[3];
    xs_state[2] ^= t;
    xs_state[3] = rotl(xs_state[3], 45);
    return result;
}

static void seed_prng(uint64_t seed) {
    sm64_state = seed;
    for (int i = 0; i < 4; i++) xs_state[i] = splitmix64();
}

/* uniform in (0,1] -- never returns exactly 0, so log() is finite */
static inline double urand(void) {
    uint64_t v;
    do { v = xoshiro_next() >> 11; } while (v == 0);   /* 53 significant bits */
    return (double)v * (1.0 / 9007199254740992.0);
}

/* standard normal via Box-Muller, used for lognormal rate assignment */
static double nrand(void) {
    double u1 = urand(), u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* Gamma(shape a, scale 1) via Marsaglia-Tsang; exact for a > 0 */
static double gamma_rand(double a) {
    if (a < 1.0) {
        /* boost: Gamma(a) = Gamma(a+1) * U^(1/a) */
        double u = urand();
        return gamma_rand(a + 1.0) * pow(u, 1.0 / a);
    }
    double d = a - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do { x = nrand(); v = 1.0 + c * x; } while (v <= 0.0);
        v = v * v * v;
        double u = urand();
        if (u < 1.0 - 0.0331 * x * x * x * x) return d * v;
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) return d * v;
    }
}

/* ------------------------------------------------------------------ */
/* Time                                                                 */
/* ------------------------------------------------------------------ */

static inline double now_mono(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* Binary min-heap over (next_time, page_index)                         */
/* ------------------------------------------------------------------ */

typedef struct { double t; uint32_t idx; } ev_t;

typedef struct {
    ev_t    *a;
    size_t   n;
    size_t   cap;
} heap_t;

static heap_t *heap_ref = NULL;   /* set in main; used by PAUSE/RESUME */

static void heap_init(heap_t *h, size_t cap) {
    h->a = (ev_t *)malloc(sizeof(ev_t) * cap);
    if (!h->a) { fprintf(stderr, "rategen: heap alloc failed\n"); exit(1); }
    h->n = 0; h->cap = cap;
}

static void heap_push(heap_t *h, double t, uint32_t idx) {
    size_t i = h->n++;
    h->a[i].t = t; h->a[i].idx = idx;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (h->a[p].t <= h->a[i].t) break;
        ev_t tmp = h->a[p]; h->a[p] = h->a[i]; h->a[i] = tmp;
        i = p;
    }
}

/* replace root with (t, idx) and sift down -- avoids pop+push churn */
static void heap_replace_root(heap_t *h, double t, uint32_t idx) {
    h->a[0].t = t; h->a[0].idx = idx;
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < h->n && h->a[l].t < h->a[m].t) m = l;
        if (r < h->n && h->a[r].t < h->a[m].t) m = r;
        if (m == i) break;
        ev_t tmp = h->a[m]; h->a[m] = h->a[i]; h->a[i] = tmp;
        i = m;
    }
}

/* ------------------------------------------------------------------ */
/* Globals                                                              */
/* ------------------------------------------------------------------ */

static volatile sig_atomic_t g_quit = 0;
static int g_paused = 0;          /* B4: quiesce access during reclaim */
static double g_pause_started = 0.0;
static double g_pause_total = 0.0;
static void on_sig(int s) { (void)s; g_quit = 1; }

static long   PAGE = 4096;
static char  *g_map = NULL;
static size_t g_npages = 0;
static double *g_lambda = NULL;      /* ground-truth rates            */
static uint64_t *g_count = NULL;     /* realized access counts        */
static double g_t0 = 0.0;            /* monotonic start               */


/* ------------------------------------------------------------------ */
/* Residency snapshot via mincore(2)                                    */
/* ------------------------------------------------------------------ */

static int write_residency(const char *path) {
    unsigned char *vec = (unsigned char *)malloc(g_npages);
    if (!vec) return -1;
    if (mincore(g_map, g_npages * (size_t)PAGE, vec) != 0) {
        fprintf(stderr, "rategen: mincore failed: %s\n", strerror(errno));
        free(vec);
        return -1;
    }
    FILE *f = fopen(path, "wb");
    if (!f) { free(vec); return -1; }
    /* header: magic, npages, monotonic time since start */
    double t = now_mono() - g_t0;
    fprintf(f, "#RHXRESID1 npages=%zu t=%.9f\n", g_npages, t);
    /* one byte per page: '1' resident, '0' not (LSB of mincore result) */
    for (size_t i = 0; i < g_npages; i++) fputc((vec[i] & 1) ? '1' : '0', f);
    fputc('\n', f);
    fclose(f);
    free(vec);
    return 0;
}

static int write_counts(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    double t = now_mono() - g_t0;
    fprintf(f, "#RHXCOUNT1 npages=%zu elapsed=%.9f paused=%.9f active=%.9f\n",
            g_npages, t, g_pause_total, t - g_pause_total);
    fprintf(f, "page\tlambda_true\tcount\n");
    for (size_t i = 0; i < g_npages; i++)
        fprintf(f, "%zu\t%.10g\t%llu\n", i, g_lambda[i],
                (unsigned long long)g_count[i]);
    fclose(f);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Rate assignment                                                      */
/* ------------------------------------------------------------------ */

typedef enum { DIST_GAMMA, DIST_UNIFORM, DIST_LOGNORMAL, DIST_DISCRETE } dist_t;

static void assign_rates(dist_t d, double p1, double p2,
                         const double *disc, size_t ndisc)
{
    for (size_t i = 0; i < g_npages; i++) {
        double lam;
        switch (d) {
        case DIST_GAMMA:      /* p1 = shape, p2 = scale                 */
            lam = gamma_rand(p1) * p2;
            break;
        case DIST_UNIFORM:    /* p1 = rate for every page (homogeneous) */
            lam = p1;
            break;
        case DIST_LOGNORMAL:  /* p1 = mu, p2 = sigma of log             */
            lam = exp(p1 + p2 * nrand());
            break;
        case DIST_DISCRETE: { /* uniform choice among listed rates      */
            size_t k = (size_t)(urand() * (double)ndisc);
            if (k >= ndisc) k = ndisc - 1;
            lam = disc[k];
            break;
        }
        default:
            lam = 1.0;
        }
        if (!(lam > 0.0) || !isfinite(lam)) lam = 1e-9;   /* guard */
        g_lambda[i] = lam;
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

static void usage(void) {
    fprintf(stderr,
"rategen -- synthetic workload with known per-page access rates\n"
"\n"
"Required:\n"
"  --backing FILE|ANON      page backing (FILE = reclaimable w/o swap)\n"
"  --path PATH              backing file path (required for FILE)\n"
"  --pages N                number of pages\n"
"  --seed S                 PRNG seed (recorded; run is reproducible)\n"
"  --truth PATH             write ground-truth rate assignment here\n"
"  --ctl PATH               control FIFO path\n"
"\n"
"Distribution (choose one):\n"
"  --gamma SHAPE SCALE      Gamma-distributed rates (heterogeneous)\n"
"  --uniform RATE           identical rate for every page (NEGATIVE CONTROL)\n"
"  --lognormal MU SIGMA     lognormal rates\n"
"  --discrete R1,R2,...     uniform choice among listed rates\n"
"\n"
"Optional:\n"
"  --duration SEC           run time (default: until SIGTERM)\n"
"  --prefault               touch every page once at startup\n"
"  --report SEC             periodic lag report to stderr\n"
"\n"
"Control FIFO commands (one per line):\n"
"  SNAPSHOT <path>          write mincore residency bitmap to <path>\n"
"  COUNTS <path>            write per-page access counts to <path>\n"
"  PAUSE                    stop all page access (quiesce for reclaim)\n"
"  RESUME                   resume access; pending events shifted, no burst\n"
"  RESETCOUNTS              zero the access counters, restart window\n"
"  QUIT\n");
}

int main(int argc, char **argv)
{
    const char *backing = NULL, *path = NULL, *truth = NULL, *ctlpath = NULL;
    size_t npages = 0;
    uint64_t seed = 0;
    dist_t dist = DIST_GAMMA;
    double p1 = 1.0, p2 = 1.0;
    double disc[64]; size_t ndisc = 0;
    double duration = -1.0, report_iv = 0.0;
    int prefault = 0, have_dist = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--backing") && i + 1 < argc) backing = argv[++i];
        else if (!strcmp(argv[i], "--path") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--pages") && i + 1 < argc) npages = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--truth") && i + 1 < argc) truth = argv[++i];
        else if (!strcmp(argv[i], "--ctl") && i + 1 < argc) ctlpath = argv[++i];
        else if (!strcmp(argv[i], "--gamma") && i + 2 < argc) {
            dist = DIST_GAMMA; p1 = atof(argv[++i]); p2 = atof(argv[++i]); have_dist = 1;
        } else if (!strcmp(argv[i], "--uniform") && i + 1 < argc) {
            dist = DIST_UNIFORM; p1 = atof(argv[++i]); have_dist = 1;
        } else if (!strcmp(argv[i], "--lognormal") && i + 2 < argc) {
            dist = DIST_LOGNORMAL; p1 = atof(argv[++i]); p2 = atof(argv[++i]); have_dist = 1;
        } else if (!strcmp(argv[i], "--discrete") && i + 1 < argc) {
            dist = DIST_DISCRETE; have_dist = 1;
            char *s = argv[++i], *tok = strtok(s, ",");
            while (tok && ndisc < 64) { disc[ndisc++] = atof(tok); tok = strtok(NULL, ","); }
        }
        else if (!strcmp(argv[i], "--duration") && i + 1 < argc) duration = atof(argv[++i]);
        else if (!strcmp(argv[i], "--report") && i + 1 < argc) report_iv = atof(argv[++i]);
        else if (!strcmp(argv[i], "--prefault")) prefault = 1;
        else if (!strcmp(argv[i], "--help")) { usage(); return 0; }
        else { fprintf(stderr, "rategen: unknown arg %s\n", argv[i]); usage(); return 2; }
    }

    if (!backing || !npages || !truth || !ctlpath || !have_dist) {
        fprintf(stderr, "rategen: missing required argument\n"); usage(); return 2;
    }
    if (!strcmp(backing, "FILE") && !path) {
        fprintf(stderr, "rategen: --backing FILE requires --path\n"); return 2;
    }

    PAGE = sysconf(_SC_PAGESIZE);
    if (PAGE <= 0) PAGE = 4096;
    g_npages = npages;
    size_t bytes = npages * (size_t)PAGE;

    seed_prng(seed);

    /* ---- create mapping ---- */
    if (!strcmp(backing, "FILE")) {
        int fd = open(path, O_RDWR | O_CREAT, 0644);
        if (fd < 0) { fprintf(stderr, "rategen: open %s: %s\n", path, strerror(errno)); return 1; }
        if (ftruncate(fd, (off_t)bytes) != 0) {
            fprintf(stderr, "rategen: ftruncate: %s\n", strerror(errno)); return 1;
        }
        /* CRITICAL: ftruncate creates a SPARSE file. Read-only faults on a hole
         * may be served by the shared zero page rather than allocating a
         * page-cache page, in which case there is nothing for reclaim to evict
         * and mincore residency becomes meaningless. Materialize every block by
         * writing real data before mapping. */
        {
            char *buf = (char *)malloc((size_t)PAGE);
            if (!buf) { fprintf(stderr, "rategen: buf alloc failed\n"); return 1; }
            memset(buf, 0xA5, (size_t)PAGE);
            if (lseek(fd, 0, SEEK_SET) < 0) {
                fprintf(stderr, "rategen: lseek: %s\n", strerror(errno)); return 1;
            }
            for (size_t i = 0; i < npages; i++) {
                ssize_t wr = write(fd, buf, (size_t)PAGE);
                if (wr != (ssize_t)PAGE) {
                    fprintf(stderr, "rategen: short write at page %zu: %s\n",
                            i, strerror(errno));
                    return 1;
                }
            }
            free(buf);
            if (fsync(fd) != 0)
                fprintf(stderr, "rategen: warning: fsync: %s\n", strerror(errno));
        }
        g_map = (char *)mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
    } else if (!strcmp(backing, "ANON")) {
        g_map = (char *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    } else {
        fprintf(stderr, "rategen: --backing must be FILE or ANON\n"); return 2;
    }
    if (g_map == MAP_FAILED) {
        fprintf(stderr, "rategen: mmap %zu bytes: %s\n", bytes, strerror(errno));
        return 1;
    }

    /* Random access pattern -- tell the kernel so it does not readahead the
     * whole region, which would defeat per-page residency control. */
    if (madvise(g_map, bytes, MADV_RANDOM) != 0)
        fprintf(stderr, "rategen: warning: madvise(MADV_RANDOM): %s\n", strerror(errno));

    g_lambda = (double *)malloc(sizeof(double) * npages);
    g_count  = (uint64_t *)calloc(npages, sizeof(uint64_t));
    if (!g_lambda || !g_count) { fprintf(stderr, "rategen: alloc failed\n"); return 1; }

    assign_rates(dist, p1, p2, disc, ndisc);

    /* ---- write ground truth BEFORE any access ---- */
    {
        FILE *f = fopen(truth, "w");
        if (!f) { fprintf(stderr, "rategen: open truth %s: %s\n", truth, strerror(errno)); return 1; }
        fprintf(f, "#RHXTRUTH1 npages=%zu seed=%llu page_size=%ld backing=%s\n",
                npages, (unsigned long long)seed, PAGE, backing);
        fprintf(f, "#dist=%d p1=%.17g p2=%.17g ndisc=%zu\n", (int)dist, p1, p2, ndisc);
        fprintf(f, "page\tlambda_true\n");
        for (size_t i = 0; i < npages; i++)
            fprintf(f, "%zu\t%.17g\n", i, g_lambda[i]);
        fclose(f);
    }

    /* ---- control FIFO ---- */
    unlink(ctlpath);
    if (mkfifo(ctlpath, 0666) != 0) {
        fprintf(stderr, "rategen: mkfifo %s: %s\n", ctlpath, strerror(errno)); return 1;
    }
    /* O_RDWR keeps the FIFO open across writer disconnects (no EOF spin) */
    int ctlfd = open(ctlpath, O_RDWR | O_NONBLOCK);
    if (ctlfd < 0) { fprintf(stderr, "rategen: open fifo: %s\n", strerror(errno)); return 1; }

    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);

    if (prefault) {
        for (size_t i = 0; i < npages; i++) {
            volatile char c = g_map[i * (size_t)PAGE];
            (void)c;
        }
    }

    /* ---- schedule initial events ---- */
    g_t0 = now_mono();
    heap_t heap; heap_init(&heap, npages); heap_ref = &heap;
    for (size_t i = 0; i < npages; i++) {
        double dt = -log(urand()) / g_lambda[i];
        heap_push(&heap, g_t0 + dt, (uint32_t)i);
    }

    fprintf(stderr, "rategen: ready pid=%d pages=%zu bytes=%zu backing=%s\n",
            (int)getpid(), npages, bytes, backing);
    fflush(stderr);

    /* ---- main loop ---- */
    char ctlbuf[4096]; size_t ctllen = 0;
    double next_report = g_t0 + (report_iv > 0 ? report_iv : 1e30);
    double max_lag = 0.0;
    uint64_t total_events = 0;

    while (!g_quit) {
        double now = now_mono();
        if (duration > 0 && now - g_t0 >= duration) break;

        /* -- drain control FIFO (non-blocking) -- */
        ssize_t r = read(ctlfd, ctlbuf + ctllen, sizeof(ctlbuf) - ctllen - 1);
        if (r > 0) {
            ctllen += (size_t)r;
            ctlbuf[ctllen] = '\0';
            char *line, *save = NULL;
            char *start = ctlbuf;
            char *nl;
            while ((nl = strchr(start, '\n')) != NULL) {
                *nl = '\0';
                line = start;
                char cmd[64] = {0}, arg[1024] = {0};
                if (sscanf(line, "%63s %1023s", cmd, arg) >= 1) {
                    if (!strcmp(cmd, "SNAPSHOT") && arg[0]) {
                        write_residency(arg);
                    } else if (!strcmp(cmd, "COUNTS") && arg[0]) {
                        write_counts(arg);
                    } else if (!strcmp(cmd, "RESETCOUNTS")) {
                        memset(g_count, 0, sizeof(uint64_t) * npages);
                        g_t0 = now_mono();
                    } else if (!strcmp(cmd, "PAUSE")) {
                        if (!g_paused) { g_paused = 1; g_pause_started = now_mono(); }
                    } else if (!strcmp(cmd, "RESUME")) {
                        if (g_paused) {
                            double pd = now_mono() - g_pause_started;
                            g_pause_total += pd;
                            /* Shift every pending event forward by the pause
                             * duration so the realized rate is unbiased: a
                             * pause must not create a burst on resume. */
                            for (size_t hi = 0; hi < heap_ref->n; hi++)
                                heap_ref->a[hi].t += pd;
                            g_paused = 0;
                        }
                    } else if (!strcmp(cmd, "QUIT")) {
                        g_quit = 1;
                    }
                }
                start = nl + 1;
            }
            size_t rem = ctllen - (size_t)(start - ctlbuf);
            memmove(ctlbuf, start, rem);
            ctllen = rem;
            (void)save;
        }

        /* -- while paused, service control only -- */
        if (g_paused) {
            struct timespec ts = {0, 5 * 1000 * 1000};   /* 5 ms */
            nanosleep(&ts, NULL);
            continue;
        }

        /* -- process due events -- */
        if (heap.n == 0) break;
        double due = heap.a[0].t;
        if (due > now) {
            double sleep_s = due - now;
            if (sleep_s > 0.0005) {
                struct timespec ts;
                /* cap sleep so control FIFO stays responsive */
                if (sleep_s > 0.05) sleep_s = 0.05;
                ts.tv_sec = (time_t)sleep_s;
                ts.tv_nsec = (long)((sleep_s - (double)ts.tv_sec) * 1e9);
                nanosleep(&ts, NULL);
            }
            continue;
        }

        double lag = now - due;
        if (lag > max_lag) max_lag = lag;

        uint32_t idx = heap.a[0].idx;
        volatile char c = g_map[(size_t)idx * (size_t)PAGE];
        (void)c;
        g_count[idx]++;
        total_events++;

        /* schedule next access for this page from the DUE time, not from now,
         * so that transient scheduling lag does not bias the realized rate */
        double dt = -log(urand()) / g_lambda[idx];
        heap_replace_root(&heap, due + dt, idx);

        if (report_iv > 0 && now >= next_report) {
            fprintf(stderr, "rategen: t=%.1f events=%llu max_lag=%.6fs\n",
                    now - g_t0, (unsigned long long)total_events, max_lag);
            fflush(stderr);
            max_lag = 0.0;
            next_report = now + report_iv;
        }
    }

    fprintf(stderr, "rategen: exiting events=%llu\n", (unsigned long long)total_events);
    close(ctlfd);
    unlink(ctlpath);
    munmap(g_map, bytes);
    free(g_lambda); free(g_count); free(heap.a);
    return 0;
}
