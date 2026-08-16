/**
 * Dependency resolution and wave scheduling for tool calls.
 *
 * The model emits a flat array of tool calls. Some of them depend on others,
 * either because the tool declared `dependsOn`, or because an argument contains
 * an explicit reference to another call's output.
 *
 * This module builds a DAG from those relationships and partitions it into
 * waves: maximal sets of calls that can execute in parallel because none of
 * them depends on another member of the same wave.
 *
 * Dependencies are never inferred heuristically. Argument-name similarity is
 * not evidence of a data dependency, and guessing produces silent ordering
 * bugs that yield plausible but incorrect results.
 */

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

/** Reference to another call's output, resolved after that call completes. */
export interface CallReference {
  $fromCall: string;
  /** Dot path into the referenced result, e.g. "user.id". Omit for whole result. */
  path?: string;
}

export interface ResolvedNode {
  call: ToolCall;
  /** Call ids this node must wait for. */
  dependencies: string[];
  /** Zero-indexed wave. All nodes in a wave may run concurrently. */
  wave: number;
  /** Where in the arguments a reference must be substituted. */
  references: Array<{ argumentPath: string[]; reference: CallReference }>;
}

export interface ExecutionPlan {
  nodes: ResolvedNode[];
  /** Nodes grouped by wave, in execution order. */
  waves: ResolvedNode[][];
  waveCount: number;
  /** Longest dependency chain length. The floor on achievable latency. */
  criticalPathLength: number;
}

export class CyclicDependencyError extends Error {
  readonly code = 'CYCLIC_DEPENDENCY';

  constructor(readonly cycle: string[]) {
    super(
      `Cyclic dependency detected: ${cycle.join(' -> ')}. ` +
      `Tool calls cannot form a cycle because no execution order satisfies them.`,
    );
    this.name = 'CyclicDependencyError';
  }
}

export class UnknownDependencyError extends Error {
  readonly code = 'UNKNOWN_DEPENDENCY';

  constructor(readonly callId: string, readonly missingId: string) {
    super(
      `Call "${callId}" references "${missingId}", which is not present in this batch. ` +
      `References may only point to calls in the same tool_calls array.`,
    );
    this.name = 'UnknownDependencyError';
  }
}

export class DependencyResolver {
  /**
   * Build an execution plan from a batch of tool calls.
   *
   * @param calls    Raw tool calls from the model
   * @param declared Map of tool name -> declared dependency tool names
   */
  resolve(
    calls: ToolCall[],
    declared: Map<string, string[]> = new Map(),
  ): ExecutionPlan {
    if (calls.length === 0) {
      return { nodes: [], waves: [], waveCount: 0, criticalPathLength: 0 };
    }

    this.assertUniqueIds(calls);

    const callIds = new Set(calls.map(c => c.id));
    const nameToIds = this.indexByName(calls);

    // Phase 1: extract dependencies from both sources
    const partial = calls.map(call => {
      const references = this.findReferences(call.arguments);

      const fromReferences = references.map(r => r.reference.$fromCall);
      for (const refId of fromReferences) {
        if (!callIds.has(refId)) {
          throw new UnknownDependencyError(call.id, refId);
        }
      }

      // Declared dependencies are by tool *name*; map to the concrete call ids
      // present in this batch. A declared dependency on a tool that wasn't
      // called in this batch is not an error: it simply has nothing to wait for.
      const fromDeclared = (declared.get(call.name) ?? [])
        .flatMap(depName => nameToIds.get(depName) ?? [])
        .filter(id => id !== call.id);

      const dependencies = [...new Set([...fromReferences, ...fromDeclared])];

      return { call, dependencies, references };
    });

    // Phase 2: detect cycles before attempting to layer
    const graph = new Map(partial.map(p => [p.call.id, p.dependencies]));
    const cycle = this.findCycle(graph);
    if (cycle) {
      throw new CyclicDependencyError(cycle);
    }

    // Phase 3: assign waves by longest-path-from-root depth
    const depths = this.computeDepths(graph);

    const nodes: ResolvedNode[] = partial.map(p => ({
      call: p.call,
      dependencies: p.dependencies,
      references: p.references,
      wave: depths.get(p.call.id)!,
    }));

    const waveCount = Math.max(...nodes.map(n => n.wave)) + 1;
    const waves: ResolvedNode[][] = Array.from({ length: waveCount }, () => []);
    for (const node of nodes) {
      waves[node.wave]!.push(node);
    }

    return {
      nodes,
      waves,
      waveCount,
      criticalPathLength: waveCount,
    };
  }

  /**
   * Substitute resolved references into a call's arguments.
   * Called immediately before execution, once producing calls have completed.
   */
  substituteReferences(
    node: ResolvedNode,
    completedResults: Map<string, unknown>,
  ): Record<string, unknown> {
    if (node.references.length === 0) {
      return node.call.arguments;
    }

    // Deep clone so the original plan stays inspectable for tracing
    const args = structuredClone(node.call.arguments);

    for (const { argumentPath, reference } of node.references) {
      if (!completedResults.has(reference.$fromCall)) {
        throw new Error(
          `Cannot substitute reference: call "${reference.$fromCall}" has no result. ` +
          `This indicates a wave scheduling bug, since dependencies must complete first.`,
        );
      }

      const sourceResult = completedResults.get(reference.$fromCall);
      const value = reference.path
        ? this.extractPath(sourceResult, reference.path)
        : sourceResult;

      this.setAtPath(args, argumentPath, value);
    }

    return args;
  }

  /**
   * Walk arguments recursively, collecting `$fromCall` references and the
   * paths at which they appear.
   */
  private findReferences(
    value: unknown,
    path: string[] = [],
  ): Array<{ argumentPath: string[]; reference: CallReference }> {
    if (value === null || typeof value !== 'object') {
      return [];
    }

    if (this.isCallReference(value)) {
      return [{ argumentPath: path, reference: value }];
    }

    if (Array.isArray(value)) {
      return value.flatMap((item, index) =>
        this.findReferences(item, [...path, String(index)]),
      );
    }

    return Object.entries(value as Record<string, unknown>).flatMap(([key, v]) =>
      this.findReferences(v, [...path, key]),
    );
  }

  private isCallReference(value: object): value is CallReference {
    return (
      '$fromCall' in value &&
      typeof (value as CallReference).$fromCall === 'string'
    );
  }

  /**
   * Iterative DFS with a three-colour marking scheme.
   * Iterative rather than recursive so a pathological graph cannot blow the
   * call stack, and because the explicit stack makes cycle reconstruction easy.
   */
  private findCycle(graph: Map<string, string[]>): string[] | null {
    const WHITE = 0, GREY = 1, BLACK = 2;
    const colour = new Map<string, number>();
    for (const id of graph.keys()) colour.set(id, WHITE);

    for (const start of graph.keys()) {
      if (colour.get(start) !== WHITE) continue;

      const stack: Array<{ id: string; path: string[] }> = [{ id: start, path: [start] }];

      while (stack.length > 0) {
        const { id, path } = stack[stack.length - 1]!;

        if (colour.get(id) === WHITE) {
          colour.set(id, GREY);
        }

        const deps = graph.get(id) ?? [];
        const unvisited = deps.find(d => colour.get(d) === WHITE);

        if (unvisited) {
          stack.push({ id: unvisited, path: [...path, unvisited] });
          continue;
        }

        // Any GREY dependency closes a cycle
        const greyDep = deps.find(d => colour.get(d) === GREY);
        if (greyDep) {
          const cycleStart = path.indexOf(greyDep);
          return cycleStart >= 0
            ? [...path.slice(cycleStart), greyDep]
            : [...path, greyDep];
        }

        colour.set(id, BLACK);
        stack.pop();
      }
    }

    return null;
  }

  /**
   * Wave = length of the longest dependency chain ending at this node.
   *
   * Using longest path rather than shortest is essential: a node must wait for
   * its slowest-to-become-available dependency, so scheduling it any earlier
   * would violate the ordering.
   *
   * Memoized DFS. Safe because cycles were already rejected.
   */
  private computeDepths(graph: Map<string, string[]>): Map<string, number> {
    const depths = new Map<string, number>();

    const depthOf = (id: string): number => {
      const cached = depths.get(id);
      if (cached !== undefined) return cached;

      const deps = graph.get(id) ?? [];
      const depth = deps.length === 0
        ? 0
        : Math.max(...deps.map(depthOf)) + 1;

      depths.set(id, depth);
      return depth;
    };

    for (const id of graph.keys()) depthOf(id);
    return depths;
  }

  private extractPath(source: unknown, path: string): unknown {
    let current: unknown = source;

    for (const segment of path.split('.')) {
      if (current === null || current === undefined) return undefined;
      if (typeof current !== 'object') return undefined;
      current = (current as Record<string, unknown>)[segment];
    }

    return current;
  }

  private setAtPath(target: Record<string, unknown>, path: string[], value: unknown): void {
    if (path.length === 0) return;

    let current: Record<string, unknown> = target;
    for (let i = 0; i < path.length - 1; i++) {
      const key = path[i]!;
      const next = current[key];
      if (next === null || typeof next !== 'object') {
        current[key] = {};
      }
      current = current[key] as Record<string, unknown>;
    }

    current[path[path.length - 1]!] = value;
  }

  private indexByName(calls: ToolCall[]): Map<string, string[]> {
    const index = new Map<string, string[]>();
    for (const call of calls) {
      const existing = index.get(call.name);
      if (existing) existing.push(call.id);
      else index.set(call.name, [call.id]);
    }
    return index;
  }

  private assertUniqueIds(calls: ToolCall[]): void {
    const seen = new Set<string>();
    for (const call of calls) {
      if (seen.has(call.id)) {
        throw new Error(
          `Duplicate tool call id "${call.id}". ` +
          `References resolve by id, so duplicates make resolution ambiguous.`,
        );
      }
      seen.add(call.id);
    }
  }
}
