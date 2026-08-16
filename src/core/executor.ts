/**
 * Tool Execution Engine: orchestrates LLM function calls.
 *
 * Problem: LLMs produce function call requests, but executing them requires:
 *   - Schema validation (is the call well-formed?)
 *   - Permission checking (is this tool allowed?)
 *   - Parallel execution (independent calls run concurrently)
 *   - Result aggregation (collect and format for the model)
 *   - Error recovery (retry transient failures, report permanent ones)
 *   - Timeout management (tools can't run forever)
 *
 * This engine handles the full lifecycle of tool execution,
 * from receiving the LLM's function call request to returning
 * structured results ready for the next model turn.
 */

import { z } from 'zod';
import type { ZodSchema } from 'zod';

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: ZodSchema;
  handler: (params: unknown, context: ExecutionContext) => Promise<unknown>;
  /** Tool-specific timeout (overrides global) */
  timeout?: number;
  /** Whether this tool can run in parallel with others */
  parallelizable?: boolean;
  /** Required permissions */
  permissions?: string[];
  /** Rate limit (calls per minute) */
  rateLimit?: number;
}

export interface FunctionCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ExecutionResult {
  callId: string;
  toolName: string;
  success: boolean;
  result?: unknown;
  error?: { code: string; message: string; retryable: boolean };
  latencyMs: number;
  metadata?: Record<string, unknown>;
}

export interface ExecutionContext {
  requestId: string;
  userId?: string;
  permissions: string[];
  deadline?: number;
  metadata: Record<string, unknown>;
}

export interface ExecutorConfig {
  /** Default timeout per tool call (ms) */
  defaultTimeout?: number;
  /** Maximum parallel tool executions */
  maxParallel?: number;
  /** Maximum total execution time for a batch */
  batchTimeout?: number;
  /** Retry configuration */
  retry?: { maxAttempts: number; baseDelay: number };
  /** Permission validator */
  checkPermissions?: (required: string[], context: ExecutionContext) => boolean;
}

interface RateLimitEntry {
  count: number;
  windowStart: number;
}

export class ToolExecutor {
  private tools: Map<string, ToolDefinition> = new Map();
  private rateLimits: Map<string, RateLimitEntry> = new Map();
  private readonly config: Required<Omit<ExecutorConfig, 'checkPermissions'>> & ExecutorConfig;

  private static readonly DEFAULTS = {
    defaultTimeout: 30_000,
    maxParallel: 5,
    batchTimeout: 60_000,
    retry: { maxAttempts: 2, baseDelay: 1000 },
  };

  constructor(config?: ExecutorConfig) {
    this.config = { ...ToolExecutor.DEFAULTS, ...config };
  }

  /**
   * Register a tool for execution.
   */
  register(tool: ToolDefinition): this {
    this.tools.set(tool.name, tool);
    return this;
  }

  /**
   * Register multiple tools at once.
   */
  registerAll(tools: ToolDefinition[]): this {
    for (const tool of tools) {
      this.register(tool);
    }
    return this;
  }

  /**
   * Execute a batch of function calls.
   * Parallelizable calls run concurrently; sequential ones run in order.
   */
  async executeBatch(
    calls: FunctionCall[],
    context: ExecutionContext,
  ): Promise<ExecutionResult[]> {
    const batchDeadline = Date.now() + this.config.batchTimeout;
    const results: ExecutionResult[] = [];

    // Partition into parallel and sequential groups
    const { parallel, sequential } = this.partitionCalls(calls);

    // Execute parallel calls concurrently
    if (parallel.length > 0) {
      const parallelResults = await this.executeParallel(parallel, context, batchDeadline);
      results.push(...parallelResults);
    }

    // Execute sequential calls in order
    for (const call of sequential) {
      if (Date.now() >= batchDeadline) {
        results.push(this.timeoutResult(call));
        continue;
      }
      const result = await this.executeSingle(call, context, batchDeadline);
      results.push(result);
    }

    return results;
  }

  /**
   * Execute a single function call with validation, permissions, and retry.
   */
  async executeSingle(
    call: FunctionCall,
    context: ExecutionContext,
    deadline?: number,
  ): Promise<ExecutionResult> {
    const startTime = performance.now();
    const tool = this.tools.get(call.name);

    // Tool not found
    if (!tool) {
      return {
        callId: call.id,
        toolName: call.name,
        success: false,
        error: { code: 'TOOL_NOT_FOUND', message: `Unknown tool: ${call.name}`, retryable: false },
        latencyMs: performance.now() - startTime,
      };
    }

    // Permission check
    if (tool.permissions?.length && this.config.checkPermissions) {
      const allowed = this.config.checkPermissions(tool.permissions, context);
      if (!allowed) {
        return {
          callId: call.id,
          toolName: call.name,
          success: false,
          error: { code: 'PERMISSION_DENIED', message: `Missing permissions for ${call.name}`, retryable: false },
          latencyMs: performance.now() - startTime,
        };
      }
    }

    // Rate limit check
    if (tool.rateLimit && !this.checkRateLimit(call.name, tool.rateLimit)) {
      return {
        callId: call.id,
        toolName: call.name,
        success: false,
        error: { code: 'RATE_LIMITED', message: `Rate limit exceeded for ${call.name}`, retryable: true },
        latencyMs: performance.now() - startTime,
      };
    }

    // Validate arguments
    const validation = tool.parameters.safeParse(call.arguments);
    if (!validation.success) {
      return {
        callId: call.id,
        toolName: call.name,
        success: false,
        error: {
          code: 'INVALID_ARGUMENTS',
          message: validation.error.issues.map(i => `${i.path.join('.')}: ${i.message}`).join('; '),
          retryable: false,
        },
        latencyMs: performance.now() - startTime,
      };
    }

    // Execute with retry
    const timeout = tool.timeout ?? this.config.defaultTimeout;
    const effectiveTimeout = deadline ? Math.min(timeout, deadline - Date.now()) : timeout;

    for (let attempt = 0; attempt <= this.config.retry.maxAttempts; attempt++) {
      try {
        const result = await Promise.race([
          tool.handler(validation.data, context),
          this.rejectAfter(effectiveTimeout),
        ]);

        return {
          callId: call.id,
          toolName: call.name,
          success: true,
          result,
          latencyMs: performance.now() - startTime,
        };
      } catch (error) {
        const err = error as Error;
        const isLastAttempt = attempt >= this.config.retry.maxAttempts;
        const isRetryable = !err.message.includes('PERMISSION') && !err.message.includes('INVALID');

        if (isLastAttempt || !isRetryable) {
          return {
            callId: call.id,
            toolName: call.name,
            success: false,
            error: { code: 'EXECUTION_FAILED', message: err.message, retryable: isRetryable },
            latencyMs: performance.now() - startTime,
          };
        }

        // Backoff before retry
        const delay = this.config.retry.baseDelay * Math.pow(2, attempt);
        await new Promise(r => setTimeout(r, delay));
      }
    }

    return this.timeoutResult(call);
  }

  /**
   * Get registered tools as JSON Schema (for LLM tool_choice).
   */
  getToolSchemas(): Array<{ name: string; description: string; parameters: Record<string, unknown> }> {
    return [...this.tools.values()].map(tool => ({
      name: tool.name,
      description: tool.description,
      parameters: this.zodToJsonSchema(tool.parameters),
    }));
  }

  private async executeParallel(
    calls: FunctionCall[],
    context: ExecutionContext,
    deadline: number,
  ): Promise<ExecutionResult[]> {
    // Respect maxParallel concurrency
    const results: ExecutionResult[] = [];
    const chunks = this.chunk(calls, this.config.maxParallel);

    for (const chunk of chunks) {
      const chunkResults = await Promise.all(
        chunk.map(call => this.executeSingle(call, context, deadline)),
      );
      results.push(...chunkResults);
    }

    return results;
  }

  private partitionCalls(calls: FunctionCall[]): {
    parallel: FunctionCall[];
    sequential: FunctionCall[];
  } {
    const parallel: FunctionCall[] = [];
    const sequential: FunctionCall[] = [];

    for (const call of calls) {
      const tool = this.tools.get(call.name);
      if (tool?.parallelizable !== false) {
        parallel.push(call);
      } else {
        sequential.push(call);
      }
    }

    return { parallel, sequential };
  }

  private checkRateLimit(toolName: string, limit: number): boolean {
    const now = Date.now();
    const entry = this.rateLimits.get(toolName);

    if (!entry || now - entry.windowStart > 60_000) {
      this.rateLimits.set(toolName, { count: 1, windowStart: now });
      return true;
    }

    if (entry.count >= limit) return false;
    entry.count++;
    return true;
  }

  private timeoutResult(call: FunctionCall): ExecutionResult {
    return {
      callId: call.id,
      toolName: call.name,
      success: false,
      error: { code: 'TIMEOUT', message: 'Execution timed out', retryable: true },
      latencyMs: 0,
    };
  }

  private rejectAfter(ms: number): Promise<never> {
    return new Promise((_, reject) =>
      setTimeout(() => reject(new Error('TIMEOUT')), ms),
    );
  }

  private zodToJsonSchema(schema: ZodSchema): Record<string, unknown> {
    try {
      if ('shape' in schema && typeof schema.shape === 'object') {
        const properties: Record<string, unknown> = {};
        for (const [key] of Object.entries(schema.shape as Record<string, unknown>)) {
          properties[key] = { type: 'string' };
        }
        return { type: 'object', properties };
      }
    } catch { /* fallback */ }
    return { type: 'object' };
  }

  private chunk<T>(arr: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < arr.length; i += size) {
      chunks.push(arr.slice(i, i + size));
    }
    return chunks;
  }
}
