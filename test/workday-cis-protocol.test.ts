import { describe, expect, test } from "vitest";
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import {
  buildCisRequest,
  cisChunkToGeneration,
  converseChunkToGeneration,
  messagesToConverse,
  messagesToGeminiContents,
  parseSseEvent,
  resolveCisDialect,
  splitSseEvents,
  toConverseInferenceConfig,
  toGeminiFunctionDeclarations,
} from "../src/agent/workday-cis-protocol.ts";

describe("messagesToGeminiContents", () => {
  test("maps system to systemInstruction and human to user content", () => {
    const { contents, systemInstruction } = messagesToGeminiContents([
      new SystemMessage("be brief"),
      new HumanMessage("hello"),
    ]);

    expect(systemInstruction).toEqual({ parts: [{ text: "be brief" }] });
    expect(contents).toEqual([{ role: "user", parts: [{ text: "hello" }] }]);
  });

  test("maps ai tool calls to functionCall and tool results to functionResponse", () => {
    const messages = [
      new HumanMessage("list files"),
      new AIMessage({
        content: "",
        tool_calls: [{ id: "call_1", name: "ls", args: { path: "." } }],
      }),
      new ToolMessage({ content: "a.txt", tool_call_id: "call_1" }),
    ];

    const { contents } = messagesToGeminiContents(messages);

    expect(contents[1]).toEqual({
      role: "model",
      parts: [{ functionCall: { name: "ls", args: { path: "." } } }],
    });
    expect(contents[2]).toEqual({
      role: "user",
      parts: [{ functionResponse: { name: "ls", response: { result: "a.txt" } } }],
    });
  });
});

describe("toGeminiFunctionDeclarations", () => {
  test("converts a tool schema and strips unsupported keywords", () => {
    const tool = {
      name: "search",
      description: "search docs",
      schema: {
        $schema: "http://json-schema.org/draft-07/schema#",
        type: "object",
        additionalProperties: false,
        properties: { q: { type: "string" } },
        required: ["q"],
      },
    };

    expect(toGeminiFunctionDeclarations([tool])).toEqual([
      {
        name: "search",
        description: "search docs",
        parameters: {
          type: "object",
          properties: { q: { type: "string" } },
          required: ["q"],
        },
      },
    ]);
  });
});

describe("buildCisRequest", () => {
  test("wraps input in target/task envelope with generationConfig", () => {
    const request = buildCisRequest(
      {
        targetProvider: "google",
        model: "gemini-x",
        taskType: "gcp-multimodal-v2",
        predictionType: undefined,
        generationConfig: { temperature: 0.2 },
      },
      [new HumanMessage("hi")],
      [{ name: "ls", description: "", parameters: { type: "object" } }],
    );

    expect(request).toEqual({
      target: { provider: "google", model: "gemini-x" },
      task: {
        type: "gcp-multimodal-v2",
        input: {
          contents: [{ role: "user", parts: [{ text: "hi" }] }],
          generationConfig: { temperature: 0.2 },
          tools: [
            { functionDeclarations: [{ name: "ls", description: "", parameters: { type: "object" } }] },
          ],
        },
      },
    });
  });

  test("includes prediction_type only when provided", () => {
    const request = buildCisRequest(
      {
        targetProvider: "google",
        model: "m",
        taskType: "t",
        predictionType: "text",
        generationConfig: {},
      },
      [new HumanMessage("hi")],
      [],
    );

    expect(request.task.prediction_type).toBe("text");
  });
});

describe("resolveCisDialect", () => {
  test("maps aws/converse task types to converse and others to gemini", () => {
    expect(resolveCisDialect("aws-converse-v1")).toBe("converse");
    expect(resolveCisDialect("AWS-Converse-V1")).toBe("converse");
    expect(resolveCisDialect("gcp-multimodal-v2")).toBe("gemini");
    expect(resolveCisDialect("something-else")).toBe("gemini");
  });
});

describe("messagesToConverse", () => {
  test("maps system, human, ai tool calls, and tool results", () => {
    const { messages, system } = messagesToConverse([
      new SystemMessage("be brief"),
      new HumanMessage("list files"),
      new AIMessage({
        content: "",
        tool_calls: [{ id: "call_1", name: "ls", args: { path: "." } }],
      }),
      new ToolMessage({ content: "a.txt", tool_call_id: "call_1" }),
    ]);

    expect(system).toEqual([{ text: "be brief" }]);
    expect(messages).toEqual([
      { role: "user", content: [{ text: "list files" }] },
      {
        role: "assistant",
        content: [
          { toolUse: { toolUseId: "call_1", name: "ls", input: { path: "." } } },
        ],
      },
      {
        role: "user",
        content: [
          { toolResult: { toolUseId: "call_1", content: [{ text: "a.txt" }] } },
        ],
      },
    ]);
  });

  test("merges parallel tool results into a single user message", () => {
    const { messages } = messagesToConverse([
      new AIMessage({
        content: "",
        tool_calls: [
          { id: "call_a", name: "read_file", args: {} },
          { id: "call_b", name: "ls", args: {} },
        ],
      }),
      new ToolMessage({ content: "a contents", tool_call_id: "call_a" }),
      new ToolMessage({ content: "b.txt", tool_call_id: "call_b" }),
    ]);

    expect(messages).toHaveLength(2);
    expect(messages[1]).toEqual({
      role: "user",
      content: [
        { toolResult: { toolUseId: "call_a", content: [{ text: "a contents" }] } },
        { toolResult: { toolUseId: "call_b", content: [{ text: "b.txt" }] } },
      ],
    });
  });
});

describe("toConverseInferenceConfig", () => {
  test("translates maxOutputTokens to maxTokens and drops topK", () => {
    expect(
      toConverseInferenceConfig({
        temperature: 0.2,
        maxOutputTokens: 8192,
        topK: 40,
        topP: 0.95,
      }),
    ).toEqual({ temperature: 0.2, maxTokens: 8192, topP: 0.95 });
  });
});

describe("buildCisRequest (converse dialect)", () => {
  test("builds a Bedrock Converse-shaped input for aws-converse-v1", () => {
    const request = buildCisRequest(
      {
        targetProvider: "aws",
        model: "anthropic.claude-opus-4-7",
        taskType: "aws-converse-v1",
        predictionType: undefined,
        generationConfig: { temperature: 0.2, maxOutputTokens: 1024 },
      },
      [new SystemMessage("be brief"), new HumanMessage("hi")],
      [{ name: "ls", description: "list", parameters: { type: "object" } }],
    );

    expect(request).toEqual({
      target: { provider: "aws", model: "anthropic.claude-opus-4-7" },
      task: {
        type: "aws-converse-v1",
        input: {
          messages: [{ role: "user", content: [{ text: "hi" }] }],
          system: [{ text: "be brief" }],
          inferenceConfig: { temperature: 0.2, maxTokens: 1024 },
          toolConfig: {
            tools: [
              {
                toolSpec: {
                  name: "ls",
                  description: "list",
                  inputSchema: { json: { type: "object" } },
                },
              },
            ],
          },
        },
      },
    });
  });
});

describe("converseChunkToGeneration", () => {
  test("maps a contentBlockDelta text event to a chunk", () => {
    const chunk = converseChunkToGeneration({
      output: { contentBlockDelta: { delta: { text: "Hello" }, contentBlockIndex: 0 } },
    });

    expect(chunk?.text).toBe("Hello");
  });

  test("maps a tool use start then input delta to tool_call_chunks", () => {
    const start = converseChunkToGeneration({
      output: {
        contentBlockStart: {
          start: { toolUse: { toolUseId: "t1", name: "ls" } },
          contentBlockIndex: 1,
        },
      },
    });
    const delta = converseChunkToGeneration({
      output: {
        contentBlockDelta: {
          delta: { toolUse: { input: '{"path":"."}' } },
          contentBlockIndex: 1,
        },
      },
    });

    if (!AIMessageChunk.isInstance(start?.message)) {
      throw new Error("expected an AIMessageChunk");
    }
    if (!AIMessageChunk.isInstance(delta?.message)) {
      throw new Error("expected an AIMessageChunk");
    }

    const [startChunk] = start.message.tool_call_chunks ?? [];
    expect(startChunk?.name).toBe("ls");
    expect(startChunk?.id).toBe("t1");
    expect(startChunk?.index).toBe(1);

    const [deltaChunk] = delta.message.tool_call_chunks ?? [];
    expect(deltaChunk?.args).toBe('{"path":"."}');
    expect(deltaChunk?.index).toBe(1);
  });

  test("maps metadata usage and messageStop finish reason", () => {
    const chunk = converseChunkToGeneration({
      output: { messageStop: { stopReason: "end_turn" } },
      metadata: {
        usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
      },
    });

    if (!AIMessageChunk.isInstance(chunk?.message)) {
      throw new Error("expected an AIMessageChunk");
    }

    expect(chunk.message.usage_metadata).toEqual({
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
    });
    expect(chunk.message.response_metadata.finishReason).toBe("end_turn");
  });

  test("maps a full non-streaming converse message", () => {
    const chunk = converseChunkToGeneration({
      output: {
        message: {
          role: "assistant",
          content: [
            { text: "done" },
            { toolUse: { toolUseId: "t2", name: "write", input: { a: 1 } } },
          ],
        },
      },
      stopReason: "tool_use",
    });

    expect(chunk?.text).toBe("done");

    if (!AIMessageChunk.isInstance(chunk?.message)) {
      throw new Error("expected an AIMessageChunk");
    }

    const [toolChunk] = chunk.message.tool_call_chunks ?? [];
    expect(toolChunk?.name).toBe("write");
    expect(toolChunk?.args).toBe('{"a":1}');
  });

  test("returns null for empty payloads", () => {
    expect(converseChunkToGeneration({})).toBeNull();
    expect(converseChunkToGeneration({ output: {} })).toBeNull();
  });
});

describe("splitSseEvents", () => {
  test("splits complete events and keeps the incomplete remainder", () => {
    const { events, rest } = splitSseEvents("event: a\ndata: 1\n\ndata: 2\n\ndata: par");

    expect(events).toEqual(["event: a\ndata: 1", "data: 2"]);
    expect(rest).toBe("data: par");
  });
});

describe("parseSseEvent", () => {
  test("parses event type and joins data lines, stripping one leading space", () => {
    expect(parseSseEvent("event: error\ndata: boom")).toEqual({
      event: "error",
      data: "boom",
    });
    expect(parseSseEvent("data: line1\ndata: line2")).toEqual({
      event: "message",
      data: "line1\nline2",
    });
  });
});

describe("cisChunkToGeneration", () => {
  test("maps text parts to a chunk", () => {
    const chunk = cisChunkToGeneration({
      output: { candidates: [{ content: { parts: [{ text: "Hello" }] } }] },
    });

    expect(chunk?.text).toBe("Hello");
  });

  test("maps functionCall parts to tool_call_chunks with stringified args", () => {
    const chunk = cisChunkToGeneration({
      output: {
        candidates: [
          { content: { parts: [{ functionCall: { name: "ls", args: { path: "." } } }] } },
        ],
      },
    });

    if (!AIMessageChunk.isInstance(chunk?.message)) {
      throw new Error("expected an AIMessageChunk");
    }

    const [toolCallChunk] = chunk.message.tool_call_chunks ?? [];
    expect(toolCallChunk?.name).toBe("ls");
    expect(toolCallChunk?.args).toBe('{"path":"."}');
    expect(toolCallChunk?.index).toBe(0);
  });

  test("returns null for empty payloads", () => {
    expect(cisChunkToGeneration({ output: { candidates: [] } })).toBeNull();
    expect(cisChunkToGeneration({})).toBeNull();
  });

  test("captures thoughtSignature on functionCall parts into additional_kwargs", () => {
    const chunk = cisChunkToGeneration({
      output: {
        candidates: [
          {
            content: {
              parts: [
                {
                  thoughtSignature: "sig-abc",
                  functionCall: { name: "ls", args: { path: "." } },
                },
              ],
            },
          },
        ],
      },
    });

    if (!AIMessageChunk.isInstance(chunk?.message)) {
      throw new Error("expected an AIMessageChunk");
    }

    const [toolCallChunk] = chunk.message.tool_call_chunks ?? [];
    const signatures = chunk.message.additional_kwargs[
      "__cis_thought_signatures"
    ] as Record<string, string>;

    expect(toolCallChunk?.id).toBeDefined();
    expect(signatures[toolCallChunk?.id ?? ""]).toBe("sig-abc");
  });
});

describe("messagesToGeminiContents thoughtSignature round-trip", () => {
  test("re-attaches stored thoughtSignature to the functionCall part", () => {
    const aiMessage = new AIMessage({
      content: "",
      tool_calls: [{ id: "call_1", name: "ls", args: { path: "." } }],
      additional_kwargs: {
        __cis_thought_signatures: { call_1: "sig-xyz" },
      },
    });

    const { contents } = messagesToGeminiContents([
      new HumanMessage("list files"),
      aiMessage,
    ]);

    expect(contents[1]).toEqual({
      role: "model",
      parts: [
        {
          functionCall: { name: "ls", args: { path: "." } },
          thoughtSignature: "sig-xyz",
        },
      ],
    });
  });

  test("omits thoughtSignature when none is stored", () => {
    const aiMessage = new AIMessage({
      content: "",
      tool_calls: [{ id: "call_2", name: "ls", args: {} }],
    });

    const { contents } = messagesToGeminiContents([aiMessage]);

    expect(contents[0]).toEqual({
      role: "model",
      parts: [{ functionCall: { name: "ls", args: {} } }],
    });
  });

  test("merges parallel tool responses into a single user content", () => {
    const messages = [
      new HumanMessage("explore"),
      new AIMessage({
        content: "",
        tool_calls: [
          { id: "call_a", name: "read_file", args: { file_path: "a" } },
          { id: "call_b", name: "ls", args: { path: "." } },
        ],
      }),
      new ToolMessage({ content: "file a contents", tool_call_id: "call_a" }),
      new ToolMessage({ content: "a.txt\nb.txt", tool_call_id: "call_b" }),
    ];

    const { contents } = messagesToGeminiContents(messages);

    // human, model(2 calls), single user turn with 2 functionResponse parts
    expect(contents).toHaveLength(3);
    expect(contents[2].role).toBe("user");
    expect(contents[2].parts).toEqual([
      {
        functionResponse: {
          name: "read_file",
          response: { result: "file a contents" },
        },
      },
      { functionResponse: { name: "ls", response: { result: "a.txt\nb.txt" } } },
    ]);
  });
});
