import { describe, it, expect } from 'vitest';
import {
  DependencyResolver,
  CyclicDependencyError,
  UnknownDependencyError,
} from '../src/planning/dependency-resolver';
import type { ToolCall } from '../src/planning/dependency-resolver';

const resolver = new DependencyResolver();

function call(id: string, name: string, args: Record<string, unknown> = {}): ToolCall {
  return { id, name, arguments: args };
}

describe('resolve: independent calls', () => {
  it('returns an empty plan for an empty batch', () => {
    const plan = resolver.resolve([]);
    expect(plan.waveCount).toBe(0);
    expect(plan.nodes).toEqual([]);
  });

  it('puts fully independent calls in a single wave', () => {
    const plan = resolver.resolve([
      call('a', 'search'),
      call('b', 'search'),
      call('c', 'search'),
    ]);

    expect(plan.waveCount).toBe(1);
    expect(plan.waves[0]).toHaveLength(3);
    expect(plan.criticalPathLength).toBe(1);
  });

  it('rejects duplicate call ids because references resolve by id', () => {
    expect(() => resolver.resolve([call('a', 'x'), call('a', 'y')])).toThrow(
      /Duplicate tool call id/,
    );
  });
});

describe('resolve: reference-derived dependencies', () => {
  it('sequences a call that consumes another call output', () => {
    const plan = resolver.resolve([
      call('fetch-user', 'getUser', { email: 'a@b.com' }),
      call('send', 'sendEmail', {
        userId: { $fromCall: 'fetch-user', path: 'id' },
      }),
    ]);

    expect(plan.waveCount).toBe(2);
    expect(plan.waves[0][0].call.id).toBe('fetch-user');
    expect(plan.waves[1][0].call.id).toBe('send');
    expect(plan.waves[1][0].dependencies).toEqual(['fetch-user']);
  });

  it('finds references nested in objects and arrays', () => {
    const plan = resolver.resolve([
      call('a', 'first'),
      call('b', 'second', {
        payload: {
          items: [{ ref: { $fromCall: 'a', path: 'value' } }],
        },
      }),
    ]);

    const node = plan.nodes.find(n => n.call.id === 'b')!;
    expect(node.references).toHaveLength(1);
    expect(node.references[0].argumentPath).toEqual(['payload', 'items', '0', 'ref']);
  });

  it('assigns waves by longest path, not shortest', () => {
    // a -> b -> c and a -> c. c must wait for b, so it belongs in wave 2.
    const plan = resolver.resolve([
      call('a', 'first'),
      call('b', 'second', { x: { $fromCall: 'a' } }),
      call('c', 'third', { x: { $fromCall: 'a' }, y: { $fromCall: 'b' } }),
    ]);

    const waveOf = (id: string) => plan.nodes.find(n => n.call.id === id)!.wave;

    expect(waveOf('a')).toBe(0);
    expect(waveOf('b')).toBe(1);
    expect(waveOf('c')).toBe(2);
  });

  it('rejects a reference to a call outside the batch', () => {
    expect(() =>
      resolver.resolve([call('b', 'second', { x: { $fromCall: 'ghost' } })]),
    ).toThrow(UnknownDependencyError);
  });
});

describe('resolve: declared dependencies', () => {
  it('applies name-based declarations to concrete call ids', () => {
    const declared = new Map([['sendEmail', ['getUser']]]);

    const plan = resolver.resolve(
      [call('u1', 'getUser'), call('s1', 'sendEmail')],
      declared,
    );

    expect(plan.waveCount).toBe(2);
    expect(plan.nodes.find(n => n.call.id === 's1')!.dependencies).toEqual(['u1']);
  });

  it('ignores a declared dependency on a tool absent from the batch', () => {
    const declared = new Map([['sendEmail', ['getUser']]]);
    const plan = resolver.resolve([call('s1', 'sendEmail')], declared);

    expect(plan.waveCount).toBe(1);
    expect(plan.nodes[0].dependencies).toEqual([]);
  });

  it('does not let a tool depend on itself via its own name', () => {
    const declared = new Map([['loop', ['loop']]]);
    const plan = resolver.resolve([call('l1', 'loop')], declared);

    expect(plan.nodes[0].dependencies).toEqual([]);
  });
});

describe('resolve: cycle detection', () => {
  it('throws on a two-node cycle', () => {
    expect(() =>
      resolver.resolve([
        call('a', 'first', { x: { $fromCall: 'b' } }),
        call('b', 'second', { x: { $fromCall: 'a' } }),
      ]),
    ).toThrow(CyclicDependencyError);
  });

  it('throws on a longer cycle and reports the path', () => {
    try {
      resolver.resolve([
        call('a', 'first', { x: { $fromCall: 'c' } }),
        call('b', 'second', { x: { $fromCall: 'a' } }),
        call('c', 'third', { x: { $fromCall: 'b' } }),
      ]);
      throw new Error('expected a CyclicDependencyError');
    } catch (error) {
      expect(error).toBeInstanceOf(CyclicDependencyError);
      expect((error as CyclicDependencyError).cycle.length).toBeGreaterThan(2);
    }
  });
});

describe('substituteReferences', () => {
  it('leaves arguments untouched when there are no references', () => {
    const plan = resolver.resolve([call('a', 'noop', { keep: 1 })]);
    const args = resolver.substituteReferences(plan.nodes[0], new Map());

    expect(args).toEqual({ keep: 1 });
  });

  it('substitutes a nested path from a completed result', () => {
    const plan = resolver.resolve([
      call('u1', 'getUser'),
      call('s1', 'sendEmail', {
        to: { $fromCall: 'u1', path: 'profile.email' },
        subject: 'hello',
      }),
    ]);

    const node = plan.nodes.find(n => n.call.id === 's1')!;
    const completed = new Map<string, unknown>([
      ['u1', { profile: { email: 'marcio@q1digital.com.br' } }],
    ]);

    expect(resolver.substituteReferences(node, completed)).toEqual({
      to: 'marcio@q1digital.com.br',
      subject: 'hello',
    });
  });

  it('substitutes the whole result when no path is given', () => {
    const plan = resolver.resolve([
      call('a', 'first'),
      call('b', 'second', { payload: { $fromCall: 'a' } }),
    ]);

    const node = plan.nodes.find(n => n.call.id === 'b')!;
    const completed = new Map<string, unknown>([['a', [1, 2, 3]]]);

    expect(resolver.substituteReferences(node, completed)).toEqual({
      payload: [1, 2, 3],
    });
  });

  it('does not mutate the original plan arguments', () => {
    const plan = resolver.resolve([
      call('a', 'first'),
      call('b', 'second', { x: { $fromCall: 'a' } }),
    ]);

    const node = plan.nodes.find(n => n.call.id === 'b')!;
    resolver.substituteReferences(node, new Map<string, unknown>([['a', 42]]));

    expect(node.call.arguments).toEqual({ x: { $fromCall: 'a' } });
  });

  it('yields undefined for a path that does not exist', () => {
    const plan = resolver.resolve([
      call('a', 'first'),
      call('b', 'second', { x: { $fromCall: 'a', path: 'missing.deep' } }),
    ]);

    const node = plan.nodes.find(n => n.call.id === 'b')!;
    const args = resolver.substituteReferences(
      node,
      new Map<string, unknown>([['a', { other: 1 }]]),
    );

    expect(args.x).toBeUndefined();
  });

  it('throws when a dependency result is missing, rather than passing undefined', () => {
    const plan = resolver.resolve([
      call('a', 'first'),
      call('b', 'second', { x: { $fromCall: 'a' } }),
    ]);

    const node = plan.nodes.find(n => n.call.id === 'b')!;

    expect(() => resolver.substituteReferences(node, new Map())).toThrow(
      /has no result/,
    );
  });
});
