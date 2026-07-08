import {
  afterEach,
  describe,
  expect,
  test,
  vi,
  type MockInstance,
} from "vitest";
import { HumanMessage } from "@langchain/core/messages";
import { ChatWorkdayCis } from "../src/agent/workday-cis-chat-model.ts";

type CisRequestBody = {
  target: { provider: string; model: string };
  task: {
    type: string;
    input: {
      tools?: {
        functionDeclarations: {
          name: string;
          description?: string;
          parameters?: Record<string, unknown>;
        }[];
      }[];
    };
  };
};

type FetchJsonInit = {
  method?: string;
  headers: Record<string, string>;
  body: string;
  signal?: AbortSignal;
};

function sseStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();

  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }

      controller.close();
    },
  });
}

function mockFetchOk(chunks: string[]): MockInstance<typeof fetch> {
  return vi.fn<typeof fetch>(() =>
    Promise.resolve(
      new Response(sseStream(chunks), {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      }),
    ),
  );
}

function getFirstFetchCall(fetchImpl: MockInstance<typeof fetch>): {
  url: string;
  init: FetchJsonInit;
} {
  const call = fetchImpl.mock.calls[0];

  if (call === undefined) {
    throw new Error("fetch was not called");
  }

  const [urlArg, init] = call;
  let url: string;

  if (typeof urlArg === "string") {
    url = urlArg;
  } else if (urlArg instanceof URL) {
    url = urlArg.href;
  } else if (
    typeof urlArg === "object" &&
    urlArg !== null &&
    "url" in urlArg &&
    typeof urlArg.url === "string"
  ) {
    url = urlArg.url;
  } else {
    url = "";
  }

  const requestInit = init ?? {};
  const body = typeof requestInit.body === "string" ? requestInit.body : "";

  return {
    url,
    init: {
      method: requestInit.method,
      headers: requestInit.headers as Record<string, string>,
      body,
      signal: requestInit.signal ?? undefined,
    },
  };
}

function parseCisRequestBody(body: string): CisRequestBody {
  return JSON.parse(body) as CisRequestBody;
}

function createModel(fetchImpl: typeof fetch): ChatWorkdayCis {
  return new ChatWorkdayCis({
    model: "gemini-x",
    baseURL: "https://host/ml/inference/cis/v1alpha1",
    apiKey: "k",
    headers: { "wd-pca-feature-key": "user" },
    query: { bypass_auth: "true" },
    targetProvider: "google",
    taskType: "gcp-multimodal-v2",
    predictionType: undefined,
    generationConfig: { temperature: 0.2 },
    fetchImpl,
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ChatWorkdayCis", () => {
  test("streams text tokens from SSE data events", async () => {
    const fetchImpl = mockFetchOk([
      'data: {"output":{"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}}\n\n',
      'data: {"output":{"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const model = createModel(fetchImpl);

    const tokens: string[] = [];
    for await (const chunk of await model.stream([new HumanMessage("hi")])) {
      tokens.push(chunk.content as string);
    }

    expect(tokens.join("")).toBe("Hello");
  });

  test("POSTs the CIS envelope to /predictions/stream with headers and query", async () => {
    const fetchImpl = mockFetchOk(["data: [DONE]\n\n"]);
    const model = createModel(fetchImpl);

    await model.invoke([new HumanMessage("hi")]);

    const { url, init } = getFirstFetchCall(fetchImpl);
    expect(url).toBe(
      "https://host/ml/inference/cis/v1alpha1/predictions/stream?bypass_auth=true",
    );
    expect(init.headers.authorization).toBe("Bearer k");
    expect(init.headers["wd-pca-feature-key"]).toBe("user");
    const body = parseCisRequestBody(init.body);
    expect(body.target).toEqual({ provider: "google", model: "gemini-x" });
    expect(body.task.type).toBe("gcp-multimodal-v2");
  });

  test("throws on an SSE error event", async () => {
    const fetchImpl = mockFetchOk(["event: error\ndata: upstream boom\n\n"]);
    const model = createModel(fetchImpl);

    await expect(model.invoke([new HumanMessage("hi")])).rejects.toThrow(
      /upstream boom/,
    );
  });

  test("throws a clear error on non-OK HTTP", async () => {
    const fetchImpl = vi.fn<typeof fetch>(() =>
      Promise.resolve(
        new Response("bad request", {
          status: 422,
          statusText: "Unprocessable",
        }),
      ),
    );
    const model = createModel(fetchImpl);

    await expect(model.invoke([new HumanMessage("hi")])).rejects.toThrow(/422/);
  });

  test("emits functionCall parts as tool calls", async () => {
    const fetchImpl = mockFetchOk([
      'data: {"output":{"candidates":[{"content":{"parts":[{"functionCall":{"name":"ls","args":{"path":"."}}}]}}]}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const model = createModel(fetchImpl);

    const result = await model.invoke([new HumanMessage("hi")]);

    expect(result.tool_calls?.[0]?.name).toBe("ls");
    expect(result.tool_calls?.[0]?.args).toEqual({ path: "." });
  });

  test("bindTools includes functionDeclarations in the CIS request body", async () => {
    const fetchImpl = mockFetchOk(["data: [DONE]\n\n"]);
    const model = createModel(fetchImpl);
    const bound = model.bindTools([
      {
        name: "ls",
        description: "List files",
        schema: {
          type: "object",
          properties: { path: { type: "string" } },
        },
      },
    ]);

    await bound.invoke([new HumanMessage("hi")]);

    const { init } = getFirstFetchCall(fetchImpl);
    const body = parseCisRequestBody(init.body);
    const declaration = body.task.input.tools?.[0]?.functionDeclarations[0];

    expect(declaration?.name).toBe("ls");
    expect(declaration?.description).toBe("List files");
    expect(declaration?.parameters).toEqual({
      type: "object",
      properties: { path: { type: "string" } },
    });
  });

  test("processes trailing SSE data when the stream closes without a final blank line", async () => {
    const fetchImpl = mockFetchOk([
      'data: {"output":{"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}}\n\n',
      'data: {"output":{"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}}',
    ]);
    const model = createModel(fetchImpl);

    const tokens: string[] = [];
    for await (const chunk of await model.stream([new HumanMessage("hi")])) {
      tokens.push(chunk.content as string);
    }

    expect(tokens.join("")).toBe("Hello");
  });

  test("populate llmOutput.tokenUsage when usageMetadata is present", async () => {
    const fetchImpl = mockFetchOk([
      'data: {"output":{"candidates":[{"content":{"parts":[{"text":"hi"}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const model = createModel(fetchImpl);

    const result = await model._generate([new HumanMessage("hi")], {});

    expect(result.llmOutput?.tokenUsage).toEqual({
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
    });
  });
});
