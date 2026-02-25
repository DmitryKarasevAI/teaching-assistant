function apiUrl(path) {
  // backend is behind /api
  return `/api${path}`;
}

function wsUrl(path) {
  const proto = (location.protocol === "https:") ? "wss" : "ws";
  return `${proto}://${location.host}/api${path}`;
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

async function ingest() {
  const tenant = document.getElementById("ing_tenant").value.trim();
  const course = document.getElementById("ing_course").value.trim();
  const source = document.getElementById("ing_source").value.trim();
  const text = document.getElementById("ing_text").value;

  const out = document.getElementById("ing_out");
  out.textContent = "Enqueueing...\n";

  const qs = new URLSearchParams({ tenant_id: tenant });
  if (course) qs.set("course_id", course);
  if (source) qs.set("source_id", source);

  const resp = await fetch(apiUrl(`/ingest/plain?${qs.toString()}`), {
    method: "POST",
    headers: { "Content-Type": "text/plain" },
    body: text
  });

  const data = await resp.json();
  out.textContent = `Queued:\n${pretty(data)}\n\nStreaming status...\n`;

  const ws = new WebSocket(wsUrl(`/ingest/ws/${data.document_id}`));
  ws.onmessage = (ev) => {
    out.textContent = `Queued:\n${pretty(data)}\n\nStatus:\n${ev.data}\n`;
  };
  ws.onerror = () => {
    out.textContent += "\nWebSocket error.\n";
  };
}

async function generate() {
  const tenant = document.getElementById("gen_tenant").value.trim();
  const course = document.getElementById("gen_course").value.trim();
  const n = parseInt(document.getElementById("gen_n").value, 10);
  const query = document.getElementById("gen_query").value.trim();

  const out = document.getElementById("gen_out");
  out.textContent = "Enqueueing...\n";

  const payload = {
    tenant_id: tenant,
    course_id: course || null,
    query,
    num_questions: n
  };

  const resp = await fetch(apiUrl(`/gen/questions/generate`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await resp.json();
  out.textContent = `Queued:\n${pretty(data)}\n\nStreaming status...\n`;

  const ws = new WebSocket(wsUrl(`/gen/ws/${data.task_id}`));
  ws.onmessage = (ev) => {
    out.textContent = `Queued:\n${pretty(data)}\n\nStatus:\n${ev.data}\n`;
  };
  ws.onerror = () => {
    out.textContent += "\nWebSocket error.\n";
  };
}

document.getElementById("btn_ingest").addEventListener("click", () => {
  ingest().catch(err => {
    document.getElementById("ing_out").textContent = `Error:\n${err?.stack || err}`;
  });
});

document.getElementById("btn_gen").addEventListener("click", () => {
  generate().catch(err => {
    document.getElementById("gen_out").textContent = `Error:\n${err?.stack || err}`;
  });
});
