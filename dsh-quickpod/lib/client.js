import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

export const DEFAULT_BASE_URL = "https://api.quickpod.org";

/** Secure API keys are prefixed qpk_ (mirrors the official QuickPod CLI). */
export function isApiKey(value) {
  return typeof value === "string" && value.trim().startsWith("qpk_");
}

function normalizeBaseUrl(raw) {
  const trimmed = String(raw ?? "").trim().replace(/\/+$/, "");
  if (!trimmed) return DEFAULT_BASE_URL;
  return /^https?:\/\//i.test(trimmed) ? trimmed : "https://" + trimmed;
}

function configFile() {
  const root = (process.env.DSH_HOME || "").trim() || homedir();
  return join(root, ".quickpod.json");
}

export function loadStoredConfig() {
  try {
    const raw = readFileSync(configFile(), "utf8");
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
  } catch {
    // Missing or unreadable config is fine: fall back to env vars / defaults.
  }
  return {};
}

export function saveStoredConfig(config) {
  const file = configFile();
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, JSON.stringify(config, null, 2) + "\n", { mode: 0o600 });
}

function errorMessage(data, rawText) {
  if (data && typeof data === "object" && !Array.isArray(data)) {
    for (const key of ["error", "details", "message"]) {
      const value = data[key];
      if (typeof value === "string" && value.trim()) return value.trim();
    }
  }
  const fallback = String(rawText ?? "").trim();
  return fallback || "no error body";
}

export class QuickPodClient {
  constructor(options = {}) {
    this.baseUrl = normalizeBaseUrl(options.baseUrl);
    this.token = String(options.token ?? "").trim();
  }

  setToken(token) {
    this.token = String(token ?? "").trim();
  }

  authHeaders() {
    if (!this.token) return {};
    if (isApiKey(this.token)) {
      return {
        "X-API-Key": this.token,
        Authorization: "ApiKey " + this.token,
      };
    }
    return { Authorization: "Bearer " + this.token };
  }

  async request(method, path, options = {}) {
    const { auth = false, query = {}, body } = options;
    let url = this.baseUrl + path;
    const qs = new URLSearchParams();
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null && value !== "") qs.set(key, String(value));
    }
    const queryString = qs.toString();
    if (queryString) url += "?" + queryString;

    const headers = { Accept: "application/json" };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (auth) {
      if (!this.token) {
        throw new Error(
          "QuickPod authentication required: call quickpod_connect first, or set QUICKPOD_API_KEY / QUICKPOD_TOKEN.",
        );
      }
      Object.assign(headers, this.authHeaders());
    }

    let response;
    try {
      response = await fetch(url, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch (cause) {
      const detail = cause && cause.message ? cause.message : String(cause);
      throw new Error("QuickPod request could not reach " + url + ": " + detail);
    }

    const raw = await response.text();
    let data = null;
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch {
        data = raw;
      }
    }

    if (!response.ok) {
      throw new Error(
        "QuickPod " + method + " " + path + " failed (HTTP " + response.status + "): " + errorMessage(data, raw),
      );
    }
    return data;
  }

  // -- typed helpers (endpoints verified against the official quickpod-cli) --

  search(kind, options = {}) {
    const occupied = !!options.occupied;
    const endpoint = kind === "cpu"
      ? (occupied ? "/notrentable_cpu" : "/rentable_cpu")
      : (occupied ? "/notrentable" : "/rentable");
    return this.request("GET", endpoint);
  }

  templates(scope, kind) {
    const table = {
      my: { gpu: ["/templates", true], cpu: ["/templates_cpu", true] },
      public: { gpu: ["/public_templates", false], cpu: ["/templates_cpu_public", false] },
      community: { gpu: ["/community_templates", false], cpu: ["/templates_cpu_community", false] },
    };
    const entry = table[scope] && table[scope][kind];
    if (!entry) throw new Error("unsupported template scope/kind: " + scope + "/" + kind);
    return this.request("GET", entry[0], { auth: entry[1] });
  }

  listPods(kind) {
    return this.request("GET", kind === "cpu" ? "/mypods_cpu" : "/mypods", { auth: true });
  }

  createPod(kind, payload) {
    const endpoint = kind === "cpu" ? "/update/createpod_cpu" : "/update/createpod";
    return this.request("POST", endpoint, { auth: true, body: payload });
  }

  podAction(kind, action, podUuid) {
    const base = {
      start: "/update/startpod",
      stop: "/update/stoppod",
      restart: "/update/restartpod",
      destroy: "/update/destroypod",
      logs: "/update/podlogs",
    }[action];
    if (!base) throw new Error("unsupported pod action: " + action);
    const endpoint = kind === "cpu" ? base + "_cpu" : base;
    return this.request("GET", endpoint, { auth: true, query: { pod_uuid: podUuid } });
  }

  login(email, password, twoFactorCode) {
    const body = { email, password };
    if (twoFactorCode) body.two_factor_code = twoFactorCode;
    return this.request("POST", "/update/auth/login", { body });
  }

  me() {
    return this.request("GET", "/update/auth/me", { auth: true });
  }
}
