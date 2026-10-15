/* Float64 CPU implementation of the existing absolute-coordinate chain.
 * No fast-math, changed body constants, force shaping, or control decisions.
 * Compile with the exact flags recorded by pole_cpu.py.
 */
#include <math.h>
#include <stddef.h>

#define MAX_POLES 10
#define MAX_DIM (MAX_POLES + 1)

static int acceleration(const double *state, double action, int n, double *out) {
    const int d = n + 1;
    double mass[MAX_DIM * MAX_DIM] = {0.0};
    double rhs[MAX_DIM] = {0.0};
    const double *theta = state + 1;
    const double *velocity = state + d;
    const double *omega = velocity + 1;
    if (!isfinite(action) || n < 2 || n > MAX_POLES) return 1;
    for (int i = 0; i < 2 * d; ++i) if (!isfinite(state[i])) return 3;
    action = fmax(-1.0, fmin(1.0, action));
    mass[0] = 1.0 + 0.1 * n + 1e-6;
    rhs[0] = 10.0 * action - 0.05 * velocity[0];
    for (int i = 0; i < n; ++i) {
        const double remaining = 0.1 * (n - i);
        const double sine = sin(theta[i]);
        const double cosine = cos(theta[i]);
        mass[i + 1] = mass[(i + 1) * d] = 0.5 * remaining * cosine;
        rhs[0] += 0.5 * remaining * sine * omega[i] * omega[i];
        rhs[i + 1] = 9.81 * 0.5 * remaining * sine - 0.05 * omega[i];
        for (int j = 0; j < n; ++j) {
            const double coupling = 0.25 * (0.1 * (n - (i > j ? i : j)));
            const double difference = theta[i] - theta[j];
            mass[(i + 1) * d + j + 1] = coupling * cos(difference) + (i == j ? 1e-6 : 0.0);
            rhs[i + 1] -= coupling * sin(difference) * omega[j] * omega[j];
        }
    }
    /* Partial-pivot Gaussian elimination solves exactly the same M a = f
     * as NumPy's LAPACK path; operation ordering need not be bitwise equal.
     */
    for (int k = 0; k < d; ++k) {
        int pivot = k;
        for (int i = k + 1; i < d; ++i)
            if (fabs(mass[i * d + k]) > fabs(mass[pivot * d + k])) pivot = i;
        if (!isfinite(mass[pivot * d + k]) || mass[pivot * d + k] == 0.0) return 2;
        if (pivot != k) {
            for (int j = k; j < d; ++j) {
                const double temp = mass[k * d + j];
                mass[k * d + j] = mass[pivot * d + j];
                mass[pivot * d + j] = temp;
            }
            const double temp = rhs[k]; rhs[k] = rhs[pivot]; rhs[pivot] = temp;
        }
        for (int i = k + 1; i < d; ++i) {
            const double factor = mass[i * d + k] / mass[k * d + k];
            for (int j = k + 1; j < d; ++j) mass[i * d + j] -= factor * mass[k * d + j];
            rhs[i] -= factor * rhs[k];
        }
    }
    for (int i = d - 1; i >= 0; --i) {
        double value = rhs[i];
        for (int j = i + 1; j < d; ++j) value -= mass[i * d + j] * out[j];
        out[i] = value / mass[i * d + i];
        if (!isfinite(out[i])) return 3;
    }
    return 0;
}

int rage_pole_acceleration(const double *state, double action, int n, double *out) {
    return acceleration(state, action, n, out);
}

int rage_pole_euler_tick(double *state, double action, int n, int substeps, double dt) {
    if (substeps < 1 || !isfinite(dt) || dt <= 0.0 || n < 2 || n > MAX_POLES) return 1;
    const int d = n + 1;
    double acc[MAX_DIM];
    for (int step = 0; step < substeps; ++step) {
        const int status = acceleration(state, action, n, acc);
        if (status != 0) return status;
        /* Existing semi-implicit / velocity-first Euler, not forward Euler. */
        for (int i = 0; i < d; ++i) state[d + i] += dt * acc[i];
        for (int i = 0; i < d; ++i) state[i] += dt * state[d + i];
    }
    for (int i = 0; i < 2 * d; ++i) if (!isfinite(state[i])) return 3;
    return 0;
}
