import { defineTool } from "@deepseek-ai/dsh-tools";
import {
  DEFAULT_BASE_URL,
  QuickPodClient,
  isApiKey,
  loadStoredConfig,
  saveStoredConfig,
} from "./client.js";

const name = "quickpod";
const inject = ["tools"];

// -- small helpers ------------------------------------------------------------

function str(value) {
  return value === undefined || value === null ? "" : String(value);
}

function num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function isTrue(value) {
  return value === true || value === 1 || value === "1" || value === "true";
}

function text(value) {
  return [{ type: "text", text: str(value) }];
}

function markdownTable(headers, rows) {
  const esc = (cell) => str(cell).replace(/\|/g, "\\|").replace(/\n/g, " ");
  const lines = [
    "| " + headers.join(" | ") + " |",
    "|" + headers.map(() => "---").join("|") + "|",
  ];
  for (const row of rows) lines.push("| " + row.map(esc).join(" | ") + " |");
  return lines.join("\n");
}

function keyValueTable(value, preferred) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [{ type: "text", text: JSON.stringify(value, null, 2) }];
  }
  const rows = [];
  const seen = new Set();
  for (const key of preferred) {
    if (key in value) {
      rows.push([key, str(value[key])]);
      seen.add(key);
    }
  }
  for (const key of Object.keys(value).sort()) {
    if (seen.has(key)) continue;
    const v = value[key];
    rows.push([key, v && typeof v === "object" ? JSON.stringify(v) : str(v)]);
  }
  return text(markdownTable(["KEY", "VALUE"], rows));
}

// -- renderers (signature is (args, value)) -----------------------------------

function renderSearch(_args, value) {
  const items = Array.isArray(value) ? value : [];
  if (items.length === 0) return text("No offers matched.");
  const rows = items.map((item) => [
    str(item.id),
    str(item.offer_name),
    str(item.gpu_type || item.cpu_name),
    str(item.num_gpus ?? item.cpus),
    str(item.hourly_cost),
    str(item.reliability),
    isTrue(item.verification) ? "yes" : "no",
    str(item.machines_id),
    str(item.geoinfo || item.geolocation || item.location),
  ]);
  return text(markdownTable(
    ["ID", "OFFER", "TYPE", "COUNT", "$/HR", "RELIABILITY", "VERIFIED", "MACHINE", "LOCATION"],
    rows,
  ));
}

function renderTemplates(_args, value) {
  const items = Array.isArray(value) ? value : [];
  if (items.length === 0) return text("No templates found.");
  const rows = items.map((item) => [
    str(item.id),
    str(item.template_name),
    str(item.template_type),
    str(item.image_path),
    isTrue(item.is_public) ? "yes" : "no",
    str(item.template_uuid),
  ]);
  return text(markdownTable(["ID", "NAME", "TYPE", "IMAGE", "PUBLIC", "TEMPLATE_UUID"], rows));
}

function renderPods(args, value) {
  const items = Array.isArray(value) ? value : [];
  const kind = args.kind === "cpu" ? "cpu" : "gpu";
  if (items.length === 0) return text("You have no " + kind + " pods.");
  const rows = items.map((item) => [
    str(item.id),
    str(item.altname || item.Names),
    str(item.pod_type || kind.toUpperCase()),
    str(item.State),
    str(item.Status),
    str(item.hourly_cost),
    str(item.pod_uuid),
    str(item.public_ipaddr),
  ]);
  return text(markdownTable(
    ["ID", "NAME", "TYPE", "STATE", "STATUS", "$/HR", "POD_UUID", "IP"],
    rows,
  ));
}

function renderLogs(_args, value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const key of ["logs", "log", "output", "message", "result"]) {
      if (typeof value[key] === "string" && value[key].trim()) return text(value[key]);
    }
  }
  return [{ type: "text", text: JSON.stringify(value, null, 2) }];
}

// -- offer filtering / sorting (mirrors the official CLI) ----------------------

// The public /rentable payload nests host metadata (reliability, verification,
// geolocation, public IP) under _machines. Flatten those to the top level and
// drop the bulky _machines/_user trees so filters, sort, render, and the
// canonical result all agree.
function flattenOffer(item) {
  if (!item || typeof item !== "object" || Array.isArray(item)) return item;
  const out = { ...item };
  const m = out._machines;
  delete out._machines;
  delete out._user;
  if (m && typeof m === "object") {
    for (const key of ["reliability", "verification", "geolocation", "geoinfo", "public_ipaddr", "latitude", "longitude", "perf_score"]) {
      if (out[key] === undefined && m[key] !== undefined) out[key] = m[key];
    }
  }
  return out;
}

function filterOffers(items, args) {
  const kind = args.kind === "cpu" ? "cpu" : "gpu";
  const countKey = kind === "cpu" ? "cpus" : "num_gpus";
  const typeFilter = str(args.gpu_type).toLowerCase();
  const location = str(args.location).toLowerCase();
  const out = [];
  for (const item of items) {
    const label = str(kind === "cpu" ? item.cpu_name || item.gpu_type : item.gpu_type).toLowerCase();
    if (typeFilter && !label.includes(typeFilter)) continue;
    const loc = (str(item.geolocation) + " " + str(item.geoinfo) + " " + str(item.location)).toLowerCase();
    if (location && !loc.includes(location)) continue;
    const hourly = num(item.hourly_cost);
    if (args.min_hourly !== undefined && args.min_hourly !== null && hourly < num(args.min_hourly)) continue;
    if (args.max_hourly !== undefined && args.max_hourly !== null && hourly > num(args.max_hourly)) continue;
    const count = Math.round(num(item[countKey]));
    if (args.min_count !== undefined && args.min_count !== null && count < num(args.min_count)) continue;
    if (args.max_count !== undefined && args.max_count !== null && count > num(args.max_count)) continue;
    if (args.min_reliability !== undefined && args.min_reliability !== null && num(item.reliability) < num(args.min_reliability)) continue;
    if (args.verified_only && !isTrue(item.verification)) continue;
    out.push(item);
  }
  return out;
}

function sortItems(items, key, desc) {
  if (!key) return items;
  return [...items].sort((a, b) => {
    const an = num(a[key]);
    const bn = num(b[key]);
    const useNumeric = an !== 0 || bn !== 0 || a[key] === "0" || b[key] === "0";
    const cmp = useNumeric
      ? an - bn
      : str(a[key]).toLowerCase().localeCompare(str(b[key]).toLowerCase());
    return desc ? -cmp : cmp;
  });
}

function findPod(items, ref) {
  const wanted = str(ref).trim().toLowerCase();
  for (const item of items) {
    for (const key of ["pod_uuid", "id", "altname", "Names"]) {
      if (str(item[key]).trim().toLowerCase() === wanted) return item;
    }
  }
  return undefined;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// -- plugin --------------------------------------------------------------------

function apply(ctx, config = {}) {
  const baseUrl = str(config.baseUrl) || process.env.QUICKPOD_BASE_URL || DEFAULT_BASE_URL;
  const stored = loadStoredConfig();
  const envToken = process.env.QUICKPOD_API_KEY || process.env.QUICKPOD_TOKEN || "";
  const client = new QuickPodClient({ baseUrl, token: envToken || stored.token || "" });

  const persistToken = () => saveStoredConfig({ baseUrl: client.baseUrl, token: client.token });

  const systemPrompt = ctx.get("systemPrompt");
  if (systemPrompt) {
    systemPrompt.section({
      name: "quickpod",
      order: 400,
      text: "QuickPod cloud tools are available: quickpod_search, quickpod_connect, quickpod_templates, quickpod_deploy, quickpod_pods, quickpod_status, quickpod_logs, quickpod_control, quickpod_wait. Before authenticated calls, set QUICKPOD_API_KEY (a qpk_ key) or run quickpod_connect. Reuse the pod_uuid returned by quickpod_deploy/quickpod_pods for quickpod_status, quickpod_logs, and quickpod_control.",
    });
  }

  ctx.tools.register(defineTool({
    name: "quickpod_search",
    description: "Search QuickPod rentable (or occupied) GPU/CPU offers. Public endpoint; no auth needed.",
    parameters: {
      kind: { type: "string", enum: ["gpu", "cpu"], required: true, description: "Offer kind." },
      gpu_type: { type: "string", description: "Filter by GPU/CPU type label, e.g. \"A100\", \"RTX 4090\"." },
      location: { type: "string", description: "Filter by location substring." },
      min_hourly: { type: "number", description: "Minimum hourly cost." },
      max_hourly: { type: "number", description: "Maximum hourly cost." },
      min_count: { type: "integer", description: "Minimum GPU/CPU count." },
      max_count: { type: "integer", description: "Maximum GPU/CPU count." },
      min_reliability: { type: "number", description: "Minimum reliability score." },
      verified_only: { type: "boolean", description: "Only show verified hosts." },
      occupied: { type: "boolean", description: "Show occupied offers instead of rentable." },
      limit: { type: "integer", description: "Max rows to return (default 25; 0 = all)." },
      sort: { type: "string", enum: ["hourly_cost", "reliability", "num_gpus", "cpus", "offer_name"], description: "Sort key." },
      desc: { type: "boolean", description: "Sort descending." },
    },
    output: { schema: { type: "json" }, render: renderSearch },
    async execute(args) {
      const raw = await client.search(args.kind, { occupied: !!args.occupied });
      const items = (Array.isArray(raw) ? raw : []).map(flattenOffer);
      const filtered = filterOffers(items, args);
      const sorted = sortItems(filtered, args.sort, !!args.desc);
      const limit = args.limit === undefined ? 25 : num(args.limit);
      return limit > 0 ? sorted.slice(0, limit) : sorted;
    },
  }));

  ctx.tools.register(defineTool({
    name: "quickpod_connect",
    description: "Connect to QuickPod: store an API key or bearer token, or log in with email/password. Verifies credentials against /update/auth/me and persists them.",
    parameters: {
      credential: { type: "string", description: "Secure API key (starts with qpk_) or bearer token to store. Takes precedence over email/password." },
      email: { type: "string", description: "QuickPod account email (for password login)." },
      password: { type: "string", description: "QuickPod account password (for password login)." },
      two_factor_code: { type: "string", description: "Optional two-factor code for password login." },
    },
    output: { schema: { type: "json" }, render: (_args, value) => keyValueTable(value, ["connected", "mode", "credential_type", "whoami"]) },
    async execute(args) {
      const credential = str(args.credential).trim();
      let token = credential;
      let mode = "credential";
      if (!token) {
        const email = str(args.email).trim();
        const password = str(args.password);
        if (!email || !password) throw new Error("Provide either credential or email + password.");
        const login = await client.login(email, password, str(args.two_factor_code).trim() || undefined);
        if (login && login.two_factor_required) {
          throw new Error(
            "Two-factor required (" + (str(login.two_factor_method) || "unknown") + "): " +
            (str(login.message) || "provide two_factor_code"),
          );
        }
        token = str(login && (login.authToken || login.token)).trim();
        if (!token) {
          throw new Error("Login did not return an auth token: " + str(login && (login.message || login.error)) || "unknown");
        }
        mode = "login";
      }
      client.setToken(token);
      persistToken();
      const whoami = await client.me();
      return { connected: true, mode, credential_type: isApiKey(token) ? "api_key" : "bearer_token", whoami };
    },
  }));

  ctx.tools.register(defineTool({
    name: "quickpod_templates",
    description: "List QuickPod templates (public, community, or your own) to obtain a template_uuid for deployment.",
    parameters: {
      scope: { type: "string", enum: ["public", "community", "my"], required: true, description: "Template scope." },
      kind: { type: "string", enum: ["gpu", "cpu"], required: true, description: "Template kind." },
    },
    output: { schema: { type: "json" }, render: renderTemplates },
    async execute(args) {
      return await client.templates(args.scope, args.kind);
    },
  }));

  ctx.tools.register(defineTool({
    name: "quickpod_deploy",
    description: "Deploy (create) a QuickPod GPU/CPU pod. Requires template_uuid (from quickpod_templates) and offer_id (from quickpod_search).",
    parameters: {
      kind: { type: "string", enum: ["gpu", "cpu"], required: true, description: "Pod kind." },
      template_uuid: { type: "string", required: true, description: "Template UUID from quickpod_templates." },
      offer_id: { type: "integer", required: true, description: "Offer ID (offers_id) from quickpod_search." },
      disk_size_gb: { type: "string", required: true, description: "Disk size in GB." },
      name: { type: "string", description: "Friendly pod name (altname)." },
      docker_options: { type: "string", description: "Extra docker options." },
      coupon_code: { type: "string", description: "Coupon code." },
      volume_id: { type: "integer", description: "Optional attached volume ID." },
    },
    output: { schema: { type: "json" }, render: (_args, value) => keyValueTable(value, ["status", "message", "pod_uuid", "public_ipaddr", "public_ipaddress", "open_port_start", "open_port_end"]) },
    async execute(args) {
      const body = {
        template_uuid: str(args.template_uuid),
        offers_id: num(args.offer_id),
        disk_size: str(args.disk_size_gb),
        docker_options: str(args.docker_options),
        altname: str(args.name),
        coupon_code: str(args.coupon_code),
      };
      if (args.volume_id !== undefined && args.volume_id !== null) body.volume_id = num(args.volume_id);
      return await client.createPod(args.kind, body);
    },
  }));

  ctx.tools.register(defineTool({
    name: "quickpod_pods",
    description: "List your QuickPod GPU or CPU pods with their current state and status.",
    parameters: {
      kind: { type: "string", enum: ["gpu", "cpu"], required: true, description: "Pod kind." },
    },
    output: { schema: { type: "json" }, render: renderPods },
    async execute(args) {
      return await client.listPods(args.kind);
    },
  }));

  ctx.tools.register(defineTool({
    name: "quickpod_status",
    description: "Get the current status of one QuickPod pod by UUID, numeric ID, or name.",
    parameters: {
      pod: { type: "string", required: true, description: "Pod UUID, numeric ID, or name." },
      kind: { type: "string", enum: ["gpu", "cpu"], required: true, description: "Pod kind." },
    },
    output: { schema: { type: "json" }, render: (_args, value) => keyValueTable(value, ["id", "pod_uuid", "altname", "Names", "State", "Status", "hourly_cost", "public_ipaddr", "open_port_start", "open_port_end", "ssh_host", "storage_volume_name", "created_at"]) },
    async execute(args) {
      const items = await client.listPods(args.kind);
      const pod = findPod(Array.isArray(items) ? items : [], args.pod);
      if (!pod) throw new Error("Pod " + str(args.pod) + " not found in your " + args.kind + " pods.");
      return pod;
    },
  }));

  ctx.tools.register(defineTool({
    name: "quickpod_logs",
    description: "Fetch logs for a QuickPod pod (e.g. a training run) by UUID.",
    parameters: {
      pod: { type: "string", required: true, description: "Pod UUID." },
      kind: { type: "string", enum: ["gpu", "cpu"], required: true, description: "Pod kind." },
    },
    output: { schema: { type: "json" }, render: renderLogs },
    async execute(args) {
      return await client.podAction(args.kind, "logs", str(args.pod));
    },
  }));

  ctx.tools.register(defineTool({
    name: "quickpod_control",
    description: "Start, stop, restart, or destroy a QuickPod pod.",
    parameters: {
      action: { type: "string", enum: ["start", "stop", "restart", "destroy"], required: true, description: "Lifecycle action." },
      pod: { type: "string", required: true, description: "Pod UUID." },
      kind: { type: "string", enum: ["gpu", "cpu"], required: true, description: "Pod kind." },
    },
    output: { schema: { type: "json" }, render: (_args, value) => keyValueTable(value, ["message", "result", "pod_uuid"]) },
    async execute(args) {
      return await client.podAction(args.kind, args.action, str(args.pod));
    },
  }));

  ctx.tools.register(defineTool({
    name: "quickpod_wait",
    description: "Poll a QuickPod pod until it reaches a target state (e.g. running) or a timeout. Use this to monitor training-session startup.",
    parameters: {
      pod: { type: "string", required: true, description: "Pod UUID, ID, or name." },
      kind: { type: "string", enum: ["gpu", "cpu"], required: true, description: "Pod kind." },
      target_state: { type: "string", description: "Target State/Status substring, case-insensitive (default \"running\")." },
      timeout_seconds: { type: "integer", description: "Max seconds to wait (default 300)." },
      interval_seconds: { type: "integer", description: "Seconds between polls (default 10)." },
    },
    output: { schema: { type: "json" }, render: (_args, value) => keyValueTable(value, ["pod_uuid", "state", "status", "target", "elapsed_seconds", "timed_out"]) },
    async execute(args, exec) {
      const target = (str(args.target_state) || "running").toLowerCase();
      const timeoutMs = (args.timeout_seconds === undefined ? 300 : num(args.timeout_seconds)) * 1000;
      const intervalMs = (args.interval_seconds === undefined ? 10 : num(args.interval_seconds)) * 1000;
      const start = Date.now();
      let last;
      for (;;) {
        if (exec && exec.signal && exec.signal.aborted) throw new Error("aborted");
        const items = await client.listPods(args.kind);
        const pod = findPod(Array.isArray(items) ? items : [], args.pod);
        if (!pod) throw new Error("Pod " + str(args.pod) + " not found in your " + args.kind + " pods.");
        last = pod;
        const state = (str(pod.State) + " " + str(pod.Status)).toLowerCase();
        if (state.includes(target)) {
          return {
            pod_uuid: str(pod.pod_uuid),
            state: str(pod.State),
            status: str(pod.Status),
            target,
            elapsed_seconds: (Date.now() - start) / 1000,
            timed_out: false,
            pod,
          };
        }
        if (Date.now() - start >= timeoutMs) {
          return {
            pod_uuid: str(pod.pod_uuid),
            state: str(pod.State),
            status: str(pod.Status),
            target,
            elapsed_seconds: (Date.now() - start) / 1000,
            timed_out: true,
            pod,
          };
        }
        await sleep(intervalMs);
      }
    },
  }));
}

export { apply, inject, name };
