/**
 * function-calling-engine: LLM tool-use orchestration.
 *
 * Three concerns, deliberately separate:
 *   DependencyResolver -> what order can these calls run in
 *   ToolExecutor       -> validate, authorize and invoke a single call
 *   RetryPolicy        -> is this failure worth retrying, and is that safe
 *
 * Each is usable standalone. ToolPipeline composes them for the common case:
 * a flat batch of calls from the model where some depend on others' output.
 */

import { ToolExecutor } from './core/executor';
import { DependencyResolver } from './planning/dependency-resolver';
import { ErrorClassifier, RetryPolicy, TerminalError } from './resilience/error-classifier';

import type {
  ToolDefinition,
  FunctionCall,
  ExecutionResult,
  ExecutionContext,
  ExecutorConfig,
} from './core/executor';
import type {
  ToolCall,
  CallReference,
  ResolvedNode,
  ExecutionPlan,
} from './planning/dependency-resolver';
import {
  CyclicDependencyError,
  UnknownDependencyError,
} from './planning/dependency-resolver';
import type {
  ErrorClass,
  ClassifiedError,
  RetryConfig,
} from './resilience/error-classifier';

export {
  ToolExecutor,
  DependencyResolver,
  ErrorClassifier,
  RetryPolicy,
  TerminalError,
  CyclicDependencyError,
  UnknownDependencyError,
};

export type {
  ToolDefinition,
  FunctionCall,
  ExecutionResult,
  ExecutionContext,
  ExecutorConfig,
  ToolCall,
  CallReference,
  ResolvedNode,
  ExecutionPlan,
  ErrorClass,
  ClassifiedError,
  RetryConfig,
};

export const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxAttempts: 3,
  baseDelayMs: 250,
  maxDelayMs: 10_000,
  jitter: true,
};

export interface ToolPipelineConfig {
  executor?: ExecutorConfig;
  retry?: RetryConfig;
  /**
   * Tool name -> tool names it must run after. Used alongside the `$fromCall`
   * references found in arguments.
   */
  declaredDependencies?: Map<string, string[]>;
  /**
   * Tools safe to repeat after an ambiguous failure. Anything absent is
   * treated as non-idempotent, which is the safe default: a timeout on an
   * unknown-idempotency write is reported rather than silently retried.
   */
  idempotentTools?: Set<string>;
}

export interface PipelineRunResult {
  results: ExecutionResult[];
  plan: ExecutionPlan;
  /** Calls skipped because something they depended on failed. */
  skipped: Array<{ callId: string; reason: string; failedDependency: string }>;
}

export class ToolPipeline {
  private readonly resolver = new DependencyResolver();
  private readonly executor: ToolExecutor;
  private readonly retry: RetryPolicy;
  private readonly declared: Map<string, string[]>;
  private readonly idempotent: Set<string>;

  constructor(config: ToolPipelineConfig = {}) {
    this.executor = new ToolExecutor(config.executor);
    this.retry = new RetryPolicy(config.retry ?? DEFAULT_RETRY_CONFIG);
    this.declared = config.declaredDependencies ?? new Map();
    this.idempotent = config.idempotentTools ?? new Set();
  }

  register(tool: ToolDefinition): this {
    this.executor.register(tool);
    return this;
  }

  registerAll(tools: ToolDefinition[]): this {
    this.executor.registerAll(tools);
    return this;
  }

  /** Schemas to hand the model, so the tool list has a single source of truth. */
  getToolSchemas(): ReturnType<ToolExecutor['getToolSchemas']> {
    return this.executor.getToolSchemas();
  }

  /**
   * Plan without executing. Useful for tracing, cost estimation, and asserting
   * in tests that a batch parallelizes the way you expect.
   */
  plan(calls: ToolCall[]): ExecutionPlan {
    return this.resolver.resolve(calls, this.declared);
  }

  /**
   * Resolve dependencies, then run wave by wave.
   *
   * Waves run sequentially relative to each other and concurrently within
   * themselves. When a call fails, its downstream dependents are skipped rather
   * than executed with a missing argument: passing `undefined` where a real id
   * was expected produces a confidently wrong result, which is worse than an
   * explicit skip the model can reason about.
   */
  async run(
    calls: ToolCall[],
    context: ExecutionContext,
  ): Promise<PipelineRunResult> {
    const plan = this.resolver.resolve(calls, this.declared);

    const results: ExecutionResult[] = [];
    const completed = new Map<string, unknown>();
    const failed = new Set<string>();
    const skipped: PipelineRunResult['skipped'] = [];

    for (const wave of plan.waves) {
      const runnable: ResolvedNode[] = [];

      for (const node of wave) {
        const brokenDependency = node.dependencies.find(id => failed.has(id));

        if (brokenDependency) {
          failed.add(node.call.id);
          skipped.push({
            callId: node.call.id,
            reason: `Dependency "${brokenDependency}" did not produce a result`,
            failedDependency: brokenDependency,
          });
          continue;
        }

        runnable.push(node);
      }

      const waveResults = await Promise.all(
        runnable.map(node => this.executeNode(node, context, completed)),
      );

      for (let i = 0; i < waveResults.length; i += 1) {
        const result = waveResults[i];
        const node = runnable[i];

        results.push(result);

        if (result.success) {
          completed.set(node.call.id, result.result);
        } else {
          failed.add(node.call.id);
        }
      }
    }

    return { results, plan, skipped };
  }

  private async executeNode(
    node: ResolvedNode,
    context: ExecutionContext,
    completed: Map<string, unknown>,
  ): Promise<ExecutionResult> {
    const args = this.resolver.substituteReferences(node, completed);

    const call: FunctionCall = {
      id: node.call.id,
      name: node.call.name,
      arguments: args,
    };

    const isIdempotent = this.idempotent.has(node.call.name);

    try {
      const { result } = await this.retry.execute(
        async () => {
          const attempt = await this.executor.executeSingle(call, context);

          // The executor reports failure in its return value rather than by
          // throwing. Rethrow so the retry policy can classify it, but keep
          // non-retriable failures terminal so they are not attempted again.
          if (!attempt.success) {
            const error = attempt.error;
            if (error && !error.retryable) {
              throw new TerminalError(error.code, error.message);
            }
            throw new Error(error?.message ?? 'Tool execution failed');
          }

          return attempt;
        },
        { idempotent: isIdempotent },
      );

      return result;
    } catch (error) {
      const classified = (error as { classified?: ClassifiedError }).classified;

      return {
        callId: node.call.id,
        toolName: node.call.name,
        success: false,
        error: {
          code: classified?.code ?? 'EXECUTION_FAILED',
          message: classified?.message ?? (error as Error).message,
          retryable: classified?.retriable ?? false,
        },
        latencyMs: 0,
        metadata: classified ? { errorClass: classified.class } : undefined,
      };
    }
  }
}
