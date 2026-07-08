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
  messagesToGeminiContents,
  parseSseEvent,
  splitSseEvents,
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
});
