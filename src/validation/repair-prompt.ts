/**
 * Repair prompt construction.
 *
 * A plain retry after a validation failure produces an identical failure: the model
 * received no new information, so it makes the same mistake. The repair prompt
 * supplies the missing information, which is the field path, the expected type, and
 * the value that was rejected.
 *
 * Repair budget is tracked separately from transport retry budget by the caller.
 * Malformed arguments and a 503 fail for unrelated reasons, and a shared budget lets
 * a flaky API consume the attempts that repair needed.
 */

export interface FieldIssue {
  /** Dot path to the offending field, e.g. "items.0.quantity". */
  path: string;
  expected: string;
  /** Serialized received value. Truncated and masked before it reaches a prompt. */
  received?: string;
  message: string;
  /** Zod-style issue code, used to pick the right guidance. */
  code?: string;
  /** Keys the schema actually accepts. Enables typo detection. */
  knownKeys?: string[];
}

export interface RepairPromptConfig {
  /** Include a compact schema summary. Defaults to true. */
  includeSchema?: boolean;
  /** Offer nearest-key suggestions for unrecognized keys. Defaults to true. */
  suggestTypos?: boolean;
  /** Max characters of a received value shown. Defaults to 60. */
  maxValueLength?: number;
}

export interface RepairPrompt {
  prompt: string;
  /** Issues that carried a typo suggestion. Worth tracing: a high rate means the
   *  schema uses names the model does not naturally produce. */
  suggestionsOffered: Array<{ path: string; from: string; to: string }>;
  issueCount: number;
}

const DEFAULT_MAX_VALUE_LENGTH = 60;

/**
 * Patterns masked before a value enters a prompt.
 *
 * A rejected argument frequently contains exactly the data that made it invalid, and
 * the repair prompt is sent to a third-party provider. Echoing it back would leak
 * user data to satisfy a validation message.
 */
const SENSITIVE_VALUE_PATTERNS: Array<{ pattern: RegExp; label: string }> = [
  { pattern: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/, label: '<email>' },
  { pattern: /\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b/, label: '<cpf>' },
  { pattern: /\b(?:\d{4}[\s-]?){3}\d{4}\b/, label: '<card-number>' },
  { pattern: /\bsk-[A-Za-z0-9]{16,}\b/, label: '<api-key>' },
  { pattern: /\beyJ[A-Za-z0-9_-]{8,}\./, label: '<jwt>' },
];

export class RepairPromptBuilder {
  private readonly config: Required<RepairPromptConfig>;

  constructor(config: RepairPromptConfig = {}) {
    this.config = {
      includeSchema: config.includeSchema ?? true,
      suggestTypos: config.suggestTypos ?? true,
      maxValueLength: config.maxValueLength ?? DEFAULT_MAX_VALUE_LENGTH,
    };
  }

  build(params: {
    toolName: string;
    issues: readonly FieldIssue[];
    schemaSummary?: Record<string, string>;
    previousAttempts?: number;
  }): RepairPrompt {
    const { toolName, issues } = params;

    if (issues.length === 0) {
      throw new Error(
        'Cannot build a repair prompt with zero issues. Asking a model to fix an ' +
          'unspecified problem produces a different arbitrary output, not a correction.',
      );
    }

    const suggestionsOffered: RepairPrompt['suggestionsOffered'] = [];
    const lines: string[] = [];

    lines.push(`Your arguments for "${toolName}" failed validation.`);

    // Escalating framing on a repeat attempt. A model that failed the same way twice
    // is usually pattern-matching the original prompt rather than reading the error.
    if (params.previousAttempts && params.previousAttempts > 0) {
      lines.push(
        `This is repair attempt ${params.previousAttempts + 1}. The previous correction ` +
          'did not resolve every issue, so read each one below individually.',
      );
    }

    lines.push('', 'Issues:');

    for (const issue of issues) {
      lines.push(this.formatIssue(issue, suggestionsOffered));
    }

    if (this.config.includeSchema && params.schemaSummary) {
      lines.push('', 'Required schema:', JSON.stringify(params.schemaSummary, null, 2));
    }

    // Closing instruction is explicit about format. Without it a model commonly
    // returns a prose apology wrapped around the JSON, which fails to parse and
    // burns another attempt on a cosmetic problem.
    lines.push(
      '',
      'Return the corrected arguments as a single JSON object.',
      'No prose, no explanation, no markdown fence.',
    );

    return {
      prompt: lines.join('\n'),
      suggestionsOffered,
      issueCount: issues.length,
    };
  }

  private formatIssue(
    issue: FieldIssue,
    suggestions: RepairPrompt['suggestionsOffered'],
  ): string {
    const parts = [`  - ${issue.path}: expected ${issue.expected}`];

    if (issue.received !== undefined) {
      parts.push(`, received ${this.safeValue(issue.received)}`);
    }

    // "Unrecognized key" alone tends to make a model DROP the field rather than fix
    // the spelling, so a required argument silently disappears on the retry. A
    // concrete alternative makes correction reliable.
    if (this.config.suggestTypos && issue.code === 'unrecognized_keys' && issue.knownKeys) {
      const offending = issue.path.split('.').pop() ?? issue.path;
      const suggestion = this.closestKey(offending, issue.knownKeys);

      if (suggestion) {
        parts.push(` (did you mean "${suggestion}"?)`);
        suggestions.push({ path: issue.path, from: offending, to: suggestion });
      } else {
        // No near match means it is not a typo but an invented field, and saying so
        // stops the model from guessing a different wrong name.
        parts.push(
          ` (not a valid field; valid fields are: ${issue.knownKeys.join(', ')})`,
        );
      }
    }

    // Type-specific guidance. Naming the mechanical fix is more reliable than
    // restating the constraint the model already violated.
    const guidance = this.guidanceFor(issue);
    if (guidance) parts.push(` — ${guidance}`);

    return parts.join('');
  }

  private guidanceFor(issue: FieldIssue): string | null {
    if (issue.expected === 'number' && issue.received?.startsWith('"')) {
      return 'send a bare number, not a quoted string';
    }
    if (issue.expected === 'boolean' && issue.received?.startsWith('"')) {
      return 'send true or false unquoted';
    }
    if (issue.expected.startsWith('one of:')) {
      return 'use one of the listed values exactly, including case';
    }
    if (issue.code === 'invalid_type' && issue.received === 'undefined') {
      return 'this field is required and was omitted';
    }
    if (issue.expected === 'array' && issue.received?.startsWith('{')) {
      return 'wrap the value in an array';
    }
    return null;
  }

  /**
   * Truncate and mask a received value.
   *
   * Masking runs before truncation on purpose: truncating first can cut a pattern in
   * half so it no longer matches, and half a credit card number is still enough to
   * be worth not sending.
   */
  private safeValue(raw: string): string {
    let value = raw;

    for (const { pattern, label } of SENSITIVE_VALUE_PATTERNS) {
      if (pattern.test(value)) {
        value = value.replace(new RegExp(pattern.source, 'g'), label);
      }
    }

    if (value.length > this.config.maxValueLength) {
      return `${value.slice(0, this.config.maxValueLength - 3)}...`;
    }

    return value;
  }

  /**
   * Nearest key by Levenshtein distance, gated on a similarity floor.
   *
   * The floor matters: suggesting an unrelated key is worse than offering no
   * suggestion, because the model will take it and produce a differently wrong call.
   */
  private closestKey(input: string, candidates: readonly string[]): string | undefined {
    if (input.length === 0 || candidates.length === 0) return undefined;

    let best: string | undefined;
    let bestDistance = Infinity;

    for (const candidate of candidates) {
      const distance = this.levenshtein(input.toLowerCase(), candidate.toLowerCase());
      if (distance < bestDistance) {
        bestDistance = distance;
        best = candidate;
      }
    }

    // Allow up to 40% of the input length, minimum 2. A short key needs a tighter
    // bound: at distance 2, "id" is equidistant from most two-letter strings.
    const threshold = Math.max(2, Math.floor(input.length * 0.4));
    return bestDistance <= threshold ? best : undefined;
  }

  /**
   * Levenshtein distance with a two-row rolling buffer.
   *
   * O(min(a,b)) space instead of the full O(a*b) matrix. Only the previous row is
   * ever needed, and allocating a full matrix for a comparison against a handful of
   * short key names is waste.
   */
  private levenshtein(a: string, b: string): number {
    if (a === b) return 0;
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;

    let previous = Array.from({ length: b.length + 1 }, (_, i) => i);
    let current = new Array<number>(b.length + 1);

    for (let i = 1; i <= a.length; i++) {
      current[0] = i;

      for (let j = 1; j <= b.length; j++) {
        const substitutionCost = a[i - 1] === b[j - 1] ? 0 : 1;
        current[j] = Math.min(
          current[j - 1]! + 1, // insertion
          previous[j]! + 1, // deletion
          previous[j - 1]! + substitutionCost, // substitution
        );
      }

      [previous, current] = [current, previous];
    }

    return previous[b.length]!;
  }
}
