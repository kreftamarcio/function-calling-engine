/**
 * Error classification and retry policy.
 *
 * The central question is not "did this fail" but "is retrying safe and likely
 * to help". Those are different questions with different answers, and
 * conflating them is how duplicate records get created.
 */

export type ErrorClass =
  | 'transient'        // Probably works on retry: network blip, 502, 503
  | 'rate_limited'     // Works after waiting, server told us how long
  | 'terminal'         // Retrying cannot help: 400, 404, validation failure
  | 'unsafe_to_retry'; // Outcome unknown on a non-idempotent operation

export interface ClassifiedError {
  class: ErrorClass;
  code: string;
  message: string;
  retriable: boolean;
  /** Server-specified delay, when provided (Retry-After). */
  retryAfterMs?: number;
  /** Preserved for logs. Never returned to the model. */
  cause?: unknown;
}

export interface RetryConfig {
  maxAttempts: number;
  baseDelayMs: number;
  maxDelayMs: number;
  jitter: boolean;
}

/** Thrown by handlers to signal a failure that must not be retried. */
export class TerminalError extends Error {
  readonly isTerminal = true;

  constructor(readonly code: string, message: string) {
    super(message);
    this.name = 'TerminalError';
  }
}

const TRANSIENT_HTTP_STATUSES = new Set([408, 425, 500, 502, 503, 504, 522, 524]);
const TERMINAL_HTTP_STATUSES = new Set([400, 401, 403, 404, 405, 409, 410, 422]);

const TRANSIENT_SYSTEM_CODES = new Set([
  'ECONNRESET',
  'ECONNREFUSED',
  'ETIMEDOUT',
  'ENOTFOUND',
  'EAI_AGAIN',
  'EPIPE',
  'ENETUNREACH',
  'EHOSTUNREACH',
]);

export class ErrorClassifier {
  /**
   * Classify a thrown value.
   *
   * @param error      Whatever the handler threw
   * @param idempotent Whether the operation is safe to repeat
   * @param timedOut   Whether this failure was a timeout
   */
  classify(error: unknown, idempotent: boolean, timedOut = false): ClassifiedError {
    // Explicit terminal signal from the handler wins over everything.
    if (error instanceof TerminalError) {
      return {
        class: 'terminal',
        code: error.code,
        message: error.message,
        retriable: false,
        cause: error,
      };
    }

    // A timeout on a non-idempotent operation is the dangerous case.
    // The request may have been fully applied server-side before the client
    // gave up waiting. Retrying could duplicate the effect, so we refuse and
    // make the ambiguity explicit rather than guessing.
    if (timedOut && !idempotent) {
      return {
        class: 'unsafe_to_retry',
        code: 'TIMEOUT_UNKNOWN_OUTCOME',
        message:
          'Operation timed out and is not idempotent. The outcome is unknown: ' +
          'it may have succeeded server-side. Not retrying automatically to ' +
          'avoid duplicating the effect. Verify state before retrying.',
        retriable: false,
        cause: error,
      };
    }

    if (timedOut) {
      return {
        class: 'transient',
        code: 'TIMEOUT',
        message: 'Operation timed out',
        retriable: true,
        cause: error,
      };
    }

    // Abort is a deliberate cancellation, not a failure to retry.
    if (this.isAbortError(error)) {
      return {
        class: 'terminal',
        code: 'ABORTED',
        message: 'Operation was cancelled',
        retriable: false,
        cause: error,
      };
    }

    const status = this.extractHttpStatus(error);
    if (status !== null) {
      return this.classifyHttpStatus(status, error, idempotent);
    }

    const systemCode = this.extractSystemCode(error);
    if (systemCode && TRANSIENT_SYSTEM_CODES.has(systemCode)) {
      // A connection-level failure on a non-idempotent write is ambiguous for
      // the same reason a timeout is: the request may have reached the server.
      // ECONNREFUSED is the exception, since it means nothing was delivered.
      if (!idempotent && systemCode !== 'ECONNREFUSED') {
        return {
          class: 'unsafe_to_retry',
          code: systemCode,
          message:
            `Connection failure (${systemCode}) on a non-idempotent operation. ` +
            'The request may have been delivered. Not retrying automatically.',
          retriable: false,
          cause: error,
        };
      }

      return {
        class: 'transient',
        code: systemCode,
        message: `Network error: ${systemCode}`,
        retriable: true,
        cause: error,
      };
    }

    // Unknown errors default to terminal. Retrying something we do not
    // understand risks repeating a side effect for no expected benefit.
    return {
      class: 'terminal',
      code: 'UNKNOWN_ERROR',
      message: error instanceof Error ? error.message : String(error),
      retriable: false,
      cause: error,
    };
  }

  private classifyHttpStatus(
    status: number,
    error: unknown,
    idempotent: boolean,
  ): ClassifiedError {
    if (status === 429) {
      return {
        class: 'rate_limited',
        code: 'RATE_LIMITED',
        message: 'Rate limit exceeded',
        retriable: true,
        retryAfterMs: this.extractRetryAfterMs(error),
        cause: error,
      };
    }

    if (TERMINAL_HTTP_STATUSES.has(status)) {
      return {
        class: 'terminal',
        code: `HTTP_${status}`,
        message: `Request failed with status ${status}`,
        retriable: false,
        cause: error,
      };
    }

    if (TRANSIENT_HTTP_STATUSES.has(status)) {
      // 5xx after a non-idempotent write: the write may have partially applied.
      // 503 and 502 are safer (usually rejected upstream), but 500 means the
      // server was already processing, so we treat it conservatively.
      if (!idempotent && status === 500) {
        return {
          class: 'unsafe_to_retry',
          code: 'HTTP_500',
          message:
            'Server error during a non-idempotent operation. State may have ' +
            'partially changed. Not retrying automatically.',
          retriable: false,
          cause: error,
        };
      }

      return {
        class: 'transient',
        code: `HTTP_${status}`,
        message: `Transient server error ${status}`,
        retriable: true,
        cause: error,
      };
    }

    return {
      class: 'terminal',
      code: `HTTP_${status}`,
      message: `Unhandled status ${status}`,
      retriable: false,
      cause: error,
    };
  }

  private isAbortError(error: unknown): boolean {
    return (
      error instanceof Error &&
      (error.name === 'AbortError' || (error as { code?: string }).code === 'ABORT_ERR')
    );
  }

  private extractHttpStatus(error: unknown): number | null {
    if (error === null || typeof error !== 'object') return null;

    const candidate = error as { status?: unknown; statusCode?: unknown; response?: { status?: unknown } };
    for (const value of [candidate.status, candidate.statusCode, candidate.response?.status]) {
      if (typeof value === 'number' && value >= 100 && value < 600) {
        return value;
      }
    }
    return null;
  }

  private extractSystemCode(error: unknown): string | null {
    if (error === null || typeof error !== 'object') return null;
    const code = (error as { code?: unknown }).code;
    return typeof code === 'string' ? code : null;
  }

  /**
   * Retry-After may be delta-seconds or an HTTP-date. Both are valid per
   * RFC 9110, and servers use both in practice.
   */
  private extractRetryAfterMs(error: unknown): number | undefined {
    if (error === null || typeof error !== 'object') return undefined;

    const headers = (error as { headers?: Record<string, unknown>; response?: { headers?: Record<string, unknown> } });
    const raw =
      headers.headers?.['retry-after'] ??
      headers.response?.headers?.['retry-after'];

    if (raw === undefined || raw === null) return undefined;

    const asString = String(raw);

    const seconds = Number(asString);
    if (Number.isFinite(seconds) && seconds >= 0) {
      return seconds * 1000;
    }

    const date = Date.parse(asString);
    if (!Number.isNaN(date)) {
      return Math.max(0, date - Date.now());
    }

    return undefined;
  }
}

export class RetryPolicy {
  constructor(
    private readonly config: RetryConfig,
    private readonly classifier = new ErrorClassifier(),
  ) {
    if (config.maxAttempts < 1) {
      throw new Error('maxAttempts must be at least 1');
    }
    if (config.baseDelayMs <= 0) {
      throw new Error('baseDelayMs must be positive');
    }
    if (config.maxDelayMs < config.baseDelayMs) {
      throw new Error('maxDelayMs must be greater than or equal to baseDelayMs');
    }
  }

  shouldRetry(classified: ClassifiedError, attemptNumber: number): boolean {
    if (!classified.retriable) return false;
    return attemptNumber < this.config.maxAttempts;
  }

  /**
   * Delay before the next attempt.
   *
   * Exponential backoff with **full jitter**: `random(0, cappedDelay)`.
   *
   * Without jitter, every concurrent client retries after the identical delay,
   * recreating the load spike that caused the failure. Full jitter spreads
   * retries uniformly across the window and is what AWS's architecture
   * guidance recommends over equal-jitter or no-jitter variants.
   *
   * A server-supplied Retry-After always wins: the server knows more about
   * its own recovery window than our backoff curve does.
   */
  delayFor(classified: ClassifiedError, attemptNumber: number): number {
    if (classified.retryAfterMs !== undefined) {
      return Math.min(classified.retryAfterMs, this.config.maxDelayMs);
    }

    const exponential = this.config.baseDelayMs * Math.pow(2, attemptNumber - 1);
    const capped = Math.min(exponential, this.config.maxDelayMs);

    return this.config.jitter ? Math.random() * capped : capped;
  }

  /**
   * Execute with retry. Attempt numbering is 1-based so log lines read
   * "attempt 1 of 3" rather than "attempt 0 of 3".
   */
  async execute<T>(
    fn: (attempt: number) => Promise<T>,
    options: { idempotent: boolean; signal?: AbortSignal },
  ): Promise<{ result: T; attempts: number }> {
    let lastClassified: ClassifiedError | null = null;

    for (let attempt = 1; attempt <= this.config.maxAttempts; attempt++) {
      if (options.signal?.aborted) {
        throw new Error('Aborted before attempt');
      }

      try {
        const result = await fn(attempt);
        return { result, attempts: attempt };
      } catch (error) {
        const timedOut = error instanceof Error && error.name === 'TimeoutError';
        lastClassified = this.classifier.classify(error, options.idempotent, timedOut);

        if (!this.shouldRetry(lastClassified, attempt)) {
          throw Object.assign(new Error(lastClassified.message), {
            classified: lastClassified,
            attempts: attempt,
          });
        }

        await this.sleep(this.delayFor(lastClassified, attempt), options.signal);
      }
    }

    throw Object.assign(
      new Error(lastClassified?.message ?? 'Retry attempts exhausted'),
      { classified: lastClassified, attempts: this.config.maxAttempts },
    );
  }

  private sleep(ms: number, signal?: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, ms);

      // Aborting during a backoff window should not wait out the delay.
      signal?.addEventListener(
        'abort',
        () => {
          clearTimeout(timer);
          reject(new Error('Aborted during retry backoff'));
        },
        { once: true },
      );
    });
  }
}
